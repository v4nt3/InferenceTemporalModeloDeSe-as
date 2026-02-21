import torch #type: ignore
import torch.nn as nn #type: ignore
import numpy as np 
from pathlib import Path
import cv2
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from tqdm import tqdm
import h5py
import gc
from torchvision.models.video import r3d_18, R3D_18_Weights 
from torchvision.models import resnet101, ResNet101_Weights
import mediapipe as mp
    

# from transformer.core.config import DataConfig
# from transformer.core.exceptions import FeatureExtractionError


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:

    features: np.ndarray
    metadata: Dict[str, Any]


class BaseFeatureExtractor(ABC):
 
    @abstractmethod
    def extract(self, video_path: Path) -> ExtractionResult:

        pass
        
    @abstractmethod
    def extract_batch(
        self,
        video_paths: List[Path],
        output_path: Path
    ) -> Dict[str, Any]:
        pass


class VisualFeatureExtractor(BaseFeatureExtractor):

    
    SUPPORTED_BACKBONES = ["resnet101", "r3d_18"]
    
    def __init__(
        self,
        backbone: str = "r3d_18",
        device: Optional[torch.device] = None,
        batch_size: int = 16
    ):
 
        self.backbone_name = backbone
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        
        # Initialize model
        self.model, self.feature_dim = self._create_model(backbone)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self._setup_transforms()
        
        logger.info(
            f"VisualFeatureExtractor initialized: backbone={backbone}, "
            f"feature_dim={self.feature_dim}, device={self.device}"
        )
        
    def _create_model(
        self,
        backbone: str
    ) -> Tuple[nn.Module, int]:
        """
        Create feature extraction model.
        
        Args:
            backbone: Backbone name
            
        Returns:
            Tuple of (model, feature_dim)
        """
        if backbone == "resnet101":
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            # Remove classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            feature_dim = 2048
            
        elif backbone == "r3d_18":
            model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            # Remove classification layer
            model.fc = nn.Identity()
            feature_dim = 512
            
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported: {self.SUPPORTED_BACKBONES}"
            )
            
        return model, feature_dim
        
    def _setup_transforms(self) -> None:
        """Setup image preprocessing transforms."""
        from torchvision import transforms
        
        if self.backbone_name == "resnet101":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # Video models expect different input
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(128),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989]
                )
            ])
            
    def _load_video_frames(
        self,
        video_path: Path,
        max_frames: Optional[int] = None
    ) -> np.ndarray:
        """
        Load frames from video file.
        
        Args:
            video_path: Path to video
            max_frames: Maximum frames to load
            
        Returns:
            Array of frames [num_frames, H, W, C]
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise FeatureExtractionError(
                f"Cannot open video: {video_path}",
                details={"path": str(video_path)}
            )
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            if max_frames and len(frames) >= max_frames:
                break
                
        cap.release()
        
        if not frames:
            raise FeatureExtractionError(
                f"No frames extracted from video: {video_path}",
                details={"path": str(video_path)}
            )
            
        return np.array(frames)
        
    def extract(self, video_path: Path) -> ExtractionResult:
        """
        Extract features from a single video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            ExtractionResult with features [num_frames, feature_dim]
        """
        frames = self._load_video_frames(video_path)
        num_frames = len(frames)
        
        if self.backbone_name == "resnet101":
            features = self._extract_frame_features(frames)
        else:
            features = self._extract_video_features(frames)

        del frames
        gc.collect()
            
        return ExtractionResult(
            features=features,
            metadata={
                "num_frames": num_frames,
                "feature_dim": self.feature_dim,
                "backbone": self.backbone_name
            }
        )
        
    def _extract_frame_features(
        self,
        frames: np.ndarray
    ) -> np.ndarray:
        """
        Extract frame-level features using 2D CNN.
        
        Args:
            frames: Video frames [num_frames, H, W, C]
            
        Returns:
            Features array [num_frames, feature_dim]
        """
        features = []
        
        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i + self.batch_size]
                
                # Transform frames
                batch_tensors = torch.stack([
                    self.transform(frame) for frame in batch_frames
                ])
                batch_tensors = batch_tensors.to(self.device)
                
                # Extract features
                batch_features = self.model(batch_tensors)
                batch_features = batch_features.squeeze(-1).squeeze(-1)
                
                features.append(batch_features.cpu().numpy())

                del batch_tensors, batch_features
                torch.cuda.empty_cache()
                
        return np.concatenate(features, axis=0)
        
    def _extract_video_features(
        self,
        frames: np.ndarray,
        clip_len: int = 16
    ) -> np.ndarray:
        """
        Extract spatio-temporal features using 3D CNN.
        
        Args:
            frames: Video frames [num_frames, H, W, C]
            clip_len: Number of frames per clip
            
        Returns:
            Features array [num_clips, feature_dim]
        """
        features = []
        num_frames = len(frames)
        
        # Process video in clips
        with torch.no_grad():
            for start in range(0, num_frames, clip_len // 2):
                end = min(start + clip_len, num_frames)
                clip = frames[start:end]
                
                # Pad if necessary
                if len(clip) < clip_len:
                    pad_len = clip_len - len(clip)
                    clip = np.concatenate([
                        clip,
                        np.repeat(clip[-1:], pad_len, axis=0)
                    ])
                    
                # Transform clip [T, H, W, C] -> [C, T, H, W]
                clip_tensors = torch.stack([
                    self.transform(frame) for frame in clip
                ])
                clip_tensors = clip_tensors.permute(1, 0, 2, 3).unsqueeze(0)
                clip_tensors = clip_tensors.to(self.device)
                
                # Extract features
                clip_features = self.model(clip_tensors)
                features.append(clip_features.cpu().numpy().squeeze())
                
        return np.array(features)
        
    def extract_batch(
        self,
        video_paths: List[Path],
        output_path: Path,
        sample_ids: Optional[List[str]] = None,
        flush_every: int = 50
    ) -> Dict[str, Any]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if sample_ids is None:
            sample_ids = [p.stem for p in video_paths]
            
        stats = {
            "total": len(video_paths),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }
        
        # --- CAMBIO: 'a' en vez de 'w' para no borrar lo existente ---
        with h5py.File(output_path, 'a') as h5f:
            existing_keys = set(h5f.keys())
            
            for idx, (path, sample_id) in enumerate(tqdm(
                zip(video_paths, sample_ids),
                total=len(video_paths),
                desc="Extracting visual features"
            )):
                # --- NUEVO: saltar si ya fue procesado ---
                if sample_id in existing_keys:
                    stats["skipped"] += 1
                    continue
                    
                try:
                    result = self.extract(path)
                    h5f.create_dataset(
                        sample_id,
                        data=result.features,
                        compression="gzip"
                    )
                    stats["success"] += 1
                    
                    # --- NUEVO: liberar memoria ---
                    del result
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # --- NUEVO: flush a disco periodicamente ---
                    if (stats["success"] % flush_every) == 0:
                        h5f.flush()
                        logger.info(
                            f"Progress: {stats['success'] + stats['skipped']}/{stats['total']} "
                            f"(skipped={stats['skipped']}, new={stats['success']})"
                        )
                    
                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append({
                        "sample_id": sample_id,
                        "error": str(e)
                    })
                    logger.warning(f"Failed to extract features for {sample_id}: {e}")
                    
        logger.info(
            f"Visual feature extraction complete: "
            f"{stats['success']} new, {stats['skipped']} skipped, "
            f"{stats['failed']} failed / {stats['total']} total"
        )
        
        return stats


class PoseFeatureExtractor(BaseFeatureExtractor):
    
    FACE_LANDMARKS_SUBSET = [
        # Jaw
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        # Eyebrows
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        # Nose
        27, 28, 29, 30, 31, 32, 33, 34, 35,
        # Eyes
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        # Mouth
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 66, 67
    ]
    
    def __init__(
        self,
        config: DataConfig,
        add_velocity: bool = True,
        add_acceleration: bool = False
    ):
            
        self.config = config
        self.add_velocity = add_velocity
        self.add_acceleration = add_acceleration
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.base_dim = self._calculate_base_dim()
        self.feature_dim = self.base_dim
        if add_velocity:
            self.feature_dim += self.base_dim
        if add_acceleration:
            self.feature_dim += self.base_dim
            
        logger.info(
            f"PoseFeatureExtractor initialized: base_dim={self.base_dim}, "
            f"feature_dim={self.feature_dim}"
        )
        
    def _calculate_base_dim(self) -> int:
        return 21 * 3 * 2 + 68 * 3 + 33 * 3
        
    def extract(self, video_path: Path) -> ExtractionResult:
        """
        Extract pose features from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            ExtractionResult with pose features
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise FeatureExtractionError(
                f"Cannot open video: {video_path}"
            )
            
        keypoints = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(frame_rgb)
            
            # Extract keypoints
            frame_keypoints = self._extract_keypoints(results, frame.shape)
            keypoints.append(frame_keypoints)
            
        cap.release()
        
        if not keypoints:
            raise FeatureExtractionError(
                f"No keypoints extracted from video: {video_path}"
            )
            
        keypoints = np.array(keypoints)
        
        # Add temporal features
        features = keypoints
        
        if self.add_velocity:
            velocity = self._compute_velocity(keypoints)
            features = np.concatenate([features, velocity], axis=-1)
            
        if self.add_acceleration:
            acceleration = self._compute_acceleration(keypoints)
            features = np.concatenate([features, acceleration], axis=-1)
            
        return ExtractionResult(
            features=features.astype(np.float32),
            metadata={
                "num_frames": len(features),
                "base_dim": self.base_dim,
                "feature_dim": self.feature_dim,
                "has_velocity": self.add_velocity,
                "has_acceleration": self.add_acceleration
            }
        )
        
    def _extract_keypoints(
        self,
        results: Any,
        frame_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Extract normalized keypoints from MediaPipe results.
        
        Args:
            results: MediaPipe Holistic results
            frame_shape: Shape of the video frame
            
        Returns:
            Keypoints array [feature_dim]
        """
        h, w = frame_shape[:2]
        keypoints = []
        
        # Left hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * (21 * 3))
            
        # Right hand
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * (21 * 3))
            
        # Face (subset)
        if results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            for idx in self.FACE_LANDMARKS_SUBSET:
                if idx < len(face_lms):
                    lm = face_lms[idx]
                    keypoints.extend([lm.x, lm.y, lm.z])
                else:
                    keypoints.extend([0.0, 0.0, 0.0])
        else:
            keypoints.extend([0.0] * (68 * 3))
            
        # Body
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * (33 * 3))
            
        return np.array(keypoints)
        
    def _compute_velocity(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute velocity (first derivative)."""
        velocity = np.zeros_like(keypoints)
        velocity[1:] = keypoints[1:] - keypoints[:-1]
        return velocity
        
    def _compute_acceleration(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute acceleration (second derivative)."""
        velocity = self._compute_velocity(keypoints)
        acceleration = np.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        return acceleration
        
    def extract_batch(
        self,
        video_paths: List[Path],
        output_path: Path,
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract pose features from multiple videos.
        
        Args:
            video_paths: List of video paths
            output_path: Output HDF5 path
            sample_ids: Sample identifiers
            
        Returns:
            Extraction statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if sample_ids is None:
            sample_ids = [p.stem for p in video_paths]
            
        stats = {
            "total": len(video_paths),
            "success": 0,
            "failed": 0,
            "errors": []
        }
        
        with h5py.File(output_path, 'w') as h5f:
            for path, sample_id in tqdm(
                zip(video_paths, sample_ids),
                total=len(video_paths),
                desc="Extracting pose features"
            ):
                try:
                    result = self.extract(path)
                    h5f.create_dataset(
                        sample_id,
                        data=result.features,
                        compression="gzip"
                    )
                    stats["success"] += 1
                    
                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append({
                        "sample_id": sample_id,
                        "error": str(e)
                    })
                    logger.warning(f"Failed to extract pose for {sample_id}: {e}")
                    
        logger.info(
            f"Pose feature extraction complete: "
            f"{stats['success']}/{stats['total']} successful"
        )
        
        return stats
        
    def close(self) -> None:
        """Release MediaPipe resources."""
        self.holistic.close()
        
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def create_feature_extractor(
    extractor_type: str,
    config: Optional[DataConfig] = None,
    **kwargs
) -> BaseFeatureExtractor:
    """
    Factory function to create feature extractors.
    
    Args:
        extractor_type: Type of extractor ("visual" or "pose")
        config: Data configuration
        **kwargs: Additional arguments
        
    Returns:
        Feature extractor instance
    """
    if extractor_type == "visual":
        return VisualFeatureExtractor(**kwargs)
    elif extractor_type == "pose":
        if config is None:
            raise ValueError("config required for pose extractor")
        return PoseFeatureExtractor(config, **kwargs)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")

