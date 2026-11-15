"""
Face Extractor: MediaPipe Face Detection with margin expansion and OpenCV fallback.
Extracts facial regions for deepfake artifact inspection.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


def sample_video_frames(video_path, max_frames=16, max_seconds=30):
    """Yield uniformly sampled RGB frames and timestamps; always close the decoder."""
    capture = cv2.VideoCapture(str(video_path))
    decoded = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0 or not np.isfinite(total) or total < 1:
            raise ValueError("Video could not be opened or has invalid frame metadata.")
        if capture.get(cv2.CAP_PROP_FRAME_WIDTH) * capture.get(cv2.CAP_PROP_FRAME_HEIGHT) > 25_000_000:
            raise ValueError("Video exceeds the 25-megapixel frame limit.")
        last = min(int(total) - 1, max(0, int(fps * max_seconds) - 1))
        for index in np.unique(np.linspace(0, last, max_frames, dtype=int)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                decoded += 1
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), round(float(index) / fps, 3)
        if not decoded:
            raise ValueError("Video contains no readable sampled frames.")
    finally:
        capture.release()


class FaceExtractor:
    def __init__(
        self,
        margin_ratio: float = 0.25,
        target_size: Tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.65,
    ):
        self.margin_ratio = margin_ratio
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold

        # Initialize MediaPipe Face Detection
        self.use_mediapipe = False
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 for full-range model
                min_detection_confidence=confidence_threshold,
            )
            self.use_mediapipe = True
        except Exception:
            # Fallback to OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_faces(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in an RGB image.
        Returns a list of dicts:
        [{"bbox": [x1, y1, x2, y2], "confidence": float, "cropped_face": np.ndarray}]
        """
        h, w, _ = rgb_image.shape
        detected_faces = []

        if self.use_mediapipe:
            results = self.detector.process(rgb_image)
            if results.detections:
                for detection in results.detections:
                    score = float(detection.score[0])
                    bb = detection.location_data.relative_bounding_box
                    x_min = int(bb.xmin * w)
                    y_min = int(bb.ymin * h)
                    box_w = int(bb.width * w)
                    box_h = int(bb.height * h)

                    # Add margin
                    margin_w = int(box_w * self.margin_ratio)
                    margin_h = int(box_h * self.margin_ratio)

                    x1 = max(0, x_min - margin_w)
                    y1 = max(0, y_min - margin_h)
                    x2 = min(w, x_min + box_w + margin_w)
                    y2 = min(h, y_min + box_h + margin_h)

                    if x2 > x1 and y2 > y1:
                        crop = rgb_image[y1:y2, x1:x2]
                        if crop.size > 0:
                            resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
                            detected_faces.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": score,
                                "cropped_face": resized,
                            })
        else:
            # Haar Cascade fallback
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            faces, _, scores = self.haar_cascade.detectMultiScale3(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), outputRejectLevels=True
            )
            for (x, y, fw, fh), score in zip(faces, scores):
                # Raw Haar scores are not probabilities; discard weak detections.
                if score < 3.0:
                    continue
                margin = int(fw * self.margin_ratio)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + fw + margin)
                y2 = min(h, y + fh + margin)

                crop = rgb_image[y1:y2, x1:x2]
                resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
                detected_faces.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.85,
                    "cropped_face": resized,
                })

        # Sort by bounding box area descending (primary subject first)
        detected_faces.sort(
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )
        return detected_faces
