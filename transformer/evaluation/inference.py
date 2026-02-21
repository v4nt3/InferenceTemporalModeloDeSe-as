import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import json
import time
from collections import deque
import mediapipe as mp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
   


@dataclass
class SignPrediction:
    """Single sign prediction."""
    label: str
    confidence: float
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0


@dataclass  
class SentencePrediction:
    """Sentence prediction."""
    signs: List[SignPrediction] = field(default_factory=list)
    sentence: str = ""
    total_frames: int = 0
    processing_time: float = 0.0


class PoseExtractorLive:
    
    FACE_LANDMARKS_SUBSET = list(range(68))
    
    def __init__(self, add_velocity: bool = True):
        
        self.add_velocity = add_velocity
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5
        )
        
        self.base_dim = 21 * 3 * 2 + 68 * 3 + 33 * 3  # 429
        self.feature_dim = self.base_dim * 2 if add_velocity else self.base_dim
        
        self._prev_keypoints = None
    
    def reset(self):
        """Reset state for new video/sequence."""
        self._prev_keypoints = None
    
    def extract_frame(self, frame_rgb: np.ndarray) -> np.ndarray:

        results = self.holistic.process(frame_rgb)
        keypoints = self._extract_keypoints(results)
        
        if self.add_velocity:
            if self._prev_keypoints is not None:
                velocity = keypoints - self._prev_keypoints
            else:
                velocity = np.zeros_like(keypoints)
            self._prev_keypoints = keypoints.copy()
            return np.concatenate([keypoints, velocity]).astype(np.float32)
        
        self._prev_keypoints = keypoints.copy()
        return keypoints.astype(np.float32)
    
    def extract_video(self, video_path: str) -> np.ndarray:
        self.reset()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        all_features = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features = self.extract_frame(frame_rgb)
            all_features.append(features)
        
        cap.release()
        self.reset()
        
        if not all_features:
            raise ValueError(f"No frames extracted from: {video_path}")
        
        return np.array(all_features)
    
    def _extract_keypoints(self, results) -> np.ndarray:
        keypoints = []
        
        # Left hand (21 * 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)
        
        # Right hand (21 * 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 63)
        
        # Face subset (68 * 3 = 204)
        if results.face_landmarks:
            face_lms = results.face_landmarks.landmark
            for idx in self.FACE_LANDMARKS_SUBSET:
                if idx < len(face_lms):
                    lm = face_lms[idx]
                    keypoints.extend([lm.x, lm.y, lm.z])
                else:
                    keypoints.extend([0.0, 0.0, 0.0])
        else:
            keypoints.extend([0.0] * 204)
        
        # Body (33 * 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * 99)
        
        return np.array(keypoints)
    
    def close(self):
        self.holistic.close()


class SignLanguageInference:
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        device: Optional[str] = None
    ):

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        if config_path:
            from transformer.core.config import Config
            self.config = Config.from_yaml(config_path)

        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        self.idx_to_label = self._load_labels(labels_path)
        
        self.pose_extractor = PoseExtractorLive(add_velocity=True)
        
        self.inf_config = self.config.inference
        self.max_seq_length = self.config.data.max_seq_length
        
        logger.info(
            f"Inference: device={self.device}, "
            f"classes={len(self.idx_to_label)}, "
            f"max_seq_length={self.max_seq_length}"
        )
    
    
    def _load_model(self, checkpoint_path: str):
        from transformer.model.transformer import create_model
        
        model = create_model(
            self.config.model,
            self.config.data,
            self.device
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Model loaded with strict=True")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}")
            logger.info("Attempting partial loading")
            
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            
            matching = model_keys & ckpt_keys
            missing = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys
            
            logger.info(
                f"Keys: matching={len(matching)}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            
            if missing:
                logger.warning(f"Missing keys (first 5): {list(missing)[:5]}")
            if unexpected:
                logger.warning(f"Unexpected keys (first 5): {list(unexpected)[:5]}")
            
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model loaded with strict=False")
        
        return model.to(self.device)
    
    def _load_labels(self, labels_path: str) -> dict:
            """
            Carga las etiquetas manejando diferentes formatos de JSON.
            """
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict) and "idx_to_label" in data:
                return {int(k): v for k, v in data["idx_to_label"].items()}
                    
            raise ValueError("Formato de labels.json no reconocido")
    
    def _prepare_features(
        self, 
        features: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        num_frames = len(features)
        
        if num_frames > self.max_seq_length:
            features = features[:self.max_seq_length]
            num_frames = self.max_seq_length
        
        if num_frames < self.max_seq_length:
            pad_len = self.max_seq_length - num_frames
            features = np.concatenate([
                features,
                np.zeros((pad_len, features.shape[-1]), dtype=np.float32)
            ])
            mask = np.concatenate([
                np.ones(num_frames, dtype=bool),
                np.zeros(pad_len, dtype=bool)
            ])
        else:
            mask = np.ones(self.max_seq_length, dtype=bool)
        
        pose_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        
        return pose_tensor, mask_tensor
    

    def _predict_window(
        self,
        features: np.ndarray
    ) -> SignPrediction:

        pose_tensor, mask_tensor = self._prepare_features(features)
        
        with torch.no_grad():
            output = self.model(
                pose_features=pose_tensor,
                attention_mask=mask_tensor
            )
        
        probs = F.softmax(output.logits, dim=-1)[0]  # [num_classes]
        
        # Top-k predictions
        top_k_probs, top_k_indices = torch.topk(
            probs, min(self.inf_config.top_k, len(probs))
        )
        
        top_k = [
            (self.idx_to_label.get(idx.item(), str(idx.item())), prob.item())
            for idx, prob in zip(top_k_indices, top_k_probs)
        ]
        
        best_idx = top_k_indices[0].item()
        best_confidence = top_k_probs[0].item()
        best_label = self.idx_to_label.get(best_idx, str(best_idx))

        if best_confidence >= self.inf_config.min_confidence:
            print("\nTop predicciones:")
            for i, (label, prob) in enumerate(top_k, 1):
                print(f"{i}. {label}: {prob:.3f}")
        
        return SignPrediction(
            label=best_label,
            confidence=best_confidence,
            top_k=top_k
        )

    def _first_valid_hand_frame(self, features: np.ndarray, eps: float = 1e-6) -> Optional[int]:

        if features is None or len(features) == 0:
            return None
        hands = features[:, :126]
        valid = np.any(np.abs(hands) > eps, axis=1)
        if not valid.any():
            return None
        return int(np.argmax(valid))
    
    def predict_sign(self, video_path: str) -> SignPrediction:
        start_time = time.time()

        features = self.pose_extractor.extract_video(video_path)
        total_frames = len(features)

        first_idx = self._first_valid_hand_frame(features)
        if first_idx is None:
            logger.warning("No hands were detected in the video. The prediction will be made on the entire video.")
            offset = 0
            features_for_pred = features
        else:
            offset = first_idx
            features_for_pred = features[first_idx:]

        prediction = self._predict_window(features_for_pred)
        prediction.start_frame = offset
        prediction.end_frame = offset + len(features_for_pred)

        elapsed = time.time() - start_time
        logger.info(
            f"sign prediction: '{prediction.label}' "
            f"(conf={prediction.confidence:.3f}, frames={len(features_for_pred)}, "
            f"time={elapsed:.2f}s, first_hand_frame={offset})"
        )

        return prediction
    
    def _window_has_valid_hands(
        self,
        window_features: np.ndarray,
        min_detection_ratio: float = 0.3,
        eps: float = 1e-4
    ) -> bool:

        if window_features is None or len(window_features) == 0:
            return False

        hands = window_features[:, :126]

        valid_frames = np.any(np.abs(hands) > eps, axis=1)

        detection_ratio = np.mean(valid_frames)

        return detection_ratio >= min_detection_ratio
    
    def predict_sentence(self, video_path: str) -> SentencePrediction:

        start_time = time.time()
        
        features = self.pose_extractor.extract_video(video_path)
        original_total_frames = len(features)

        first_idx = self._first_valid_hand_frame(features)

        if first_idx is not None:
            features = features[first_idx:]
            logger.info(f"First frame with hands: {first_idx}")
        else:
            first_idx = 0

        total_frames = len(features)
        
        window_size = self.inf_config.window_size
        stride = self.inf_config.window_stride
        min_confidence = self.inf_config.min_confidence
        
        raw_predictions = []
        
        for start in range(0, total_frames, stride):
            end = min(start + window_size, total_frames)
            window_features = features[start:end]
            
            if len(window_features) < self.config.data.min_seq_length:
                continue
            
            if not self._window_has_valid_hands(window_features):
                continue
            
            prediction = self._predict_window(window_features)
            
            prediction.start_frame = start + first_idx
            prediction.end_frame = end + first_idx
            
            if prediction.confidence >= min_confidence:
                raw_predictions.append(prediction)
        
        
        if self.inf_config.merge_duplicates:
            merged = self._merge_predictions(raw_predictions)
        else:
            merged = raw_predictions
        
        sentence = " ".join([p.label for p in merged])
        
        elapsed = time.time() - start_time
        
        result = SentencePrediction(
            signs=merged,
            sentence=sentence,
            total_frames=original_total_frames,
            processing_time=elapsed
        )
        
        logger.info(
            f"Sentence prediction: '{sentence}' "
            f"({len(merged)} signs, {original_total_frames} frames, {elapsed:.2f}s)"
        )
        
        return result

 
    def _merge_predictions(
        self, predictions: List[SignPrediction]
    ) -> List[SignPrediction]:

        if not predictions:
            return []
        
        merged = [predictions[0]]
        
        for pred in predictions[1:]:
            if pred.label == merged[-1].label:

                if pred.confidence > merged[-1].confidence:
                    merged[-1] = SignPrediction(
                        label=pred.label,
                        confidence=pred.confidence,
                        top_k=pred.top_k,
                        start_frame=merged[-1].start_frame,
                        end_frame=pred.end_frame
                    )
                else:
                    merged[-1].end_frame = pred.end_frame
            else:
                merged.append(pred)
        
        return merged
    
    
    def run_camera(
        self,
        camera_index: int = 0,
        display: bool = True,
        callback: Optional[Any] = None
    ):

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_index}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or self.inf_config.camera_fps
        buffer_size = int(fps * self.inf_config.camera_buffer_seconds)
        predict_interval = self.inf_config.window_stride
        
        feature_buffer = deque(maxlen=buffer_size)
        frame_count = 0
        current_prediction = None
        
        self.pose_extractor.reset()
        
        logger.info(
            f"Camera started: fps={fps:.0f}, buffer={buffer_size} frames, "
            f"predict_every={predict_interval} frames"
        )
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                features = self.pose_extractor.extract_frame(frame_rgb)
                feature_buffer.append(features)
                frame_count += 1
                
                if (frame_count % predict_interval == 0 and 
                    len(feature_buffer) >= self.config.data.min_seq_length):
                    
                    buffer_array = np.array(list(feature_buffer))
                    prediction = self._predict_window(buffer_array)
                    
                    if prediction.confidence >= self.inf_config.min_confidence:
                        current_prediction = prediction
                        
                        if callback:
                            callback(prediction)
                
                # Display
                if display and current_prediction:
                    self._draw_prediction(frame, current_prediction)
                    cv2.imshow("Sign Language Recognition", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                elif display:
                    cv2.imshow("Sign Language Recognition", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self.pose_extractor.reset()
    
    def _draw_prediction(
        self, 
        frame: np.ndarray, 
        prediction: SignPrediction
    ) -> None:
        """Draw prediction overlay on frame."""
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (0, h - 100), (w, h), (0, 0, 0), -1)
        
        text = f"{prediction.label} ({prediction.confidence:.1%})"
        cv2.putText(
            frame, text, (20, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
        )
        
        top3_text = " | ".join(
            f"{label}: {conf:.1%}" 
            for label, conf in prediction.top_k[:3]
        )
        cv2.putText(
            frame, top3_text, (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
    
    def close(self):
        self.pose_extractor.close()



def predict_sign(
    video_path: str,
    checkpoint_path: str,
    config_path: str = None,
    labels_path: str = None
) -> SignPrediction:
    """Quick function to predict a single sign sign."""
    engine = SignLanguageInference(checkpoint_path, config_path, labels_path)
    result = engine.predict_sign(video_path)
    engine.close()
    return result


def predict_sentence(
    video_path: str,
    checkpoint_path: str,
    config_path: str = None,
    labels_path: str = None
) -> SentencePrediction:
    """Quick function to predict a sentence."""
    engine = SignLanguageInference(checkpoint_path, config_path, labels_path)
    result = engine.predict_sentence(video_path)
    engine.close()
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sign Language Inference")
    parser.add_argument("--mode", choices=["sign", "sentence", "camera"], required=True)
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--camera", type=int, default=0)
    
    args = parser.parse_args()
    
    engine = SignLanguageInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        labels_path=args.labels,
        device=args.device
    )
    
    if args.mode == "sign":
        if not args.video:
            parser.error("--video")
        result = engine.predict_sign(args.video)
        print(f"\nPrediction: {result.label}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Top-{len(result.top_k)}:")
        for label, conf in result.top_k:
            print(f"  {label}: {conf:.1%}")
    
    elif args.mode == "sentence":
        if not args.video:
            parser.error("--video")
        result = engine.predict_sentence(args.video)
        print(f"\nSentence: {result.sentence}")
        print(f"Signs detected: {len(result.signs)}")
        for i, sign in enumerate(result.signs):
            print(f"  [{sign.start_frame}-{sign.end_frame}] "
                  f"{sign.label} ({sign.confidence:.1%})")
    
    elif args.mode == "camera":
        print("Starting camera Press 'q' to quit")
        engine.run_camera(camera_index=args.camera)
    
    engine.close()
