import threading
import queue
import math
from collections import deque
from typing import Dict
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

class AsyncEmotionDetector:
    def __init__(
        self,
        buffer_size: int = 8,      # Reduced buffer for faster response
        skip_rate: int = 10,       # Process 1 out of 10 frames (better performance)
        min_confidence: float = 0.3 # Lower confidence threshold for faster detection
    ):
        self.skip_rate = skip_rate
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0, 
            min_detection_confidence=min_confidence
        )

        # Pre-load DeepFace (warmup)
        print("[AsyncEmotion] Warming up DeepFace...")
        try:
            dummy = np.zeros((48, 48, 3), dtype=np.uint8)
            DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, detector_backend='skip', silent=True)
        except:
            pass

        self.frame_queue = queue.Queue(maxsize=1)  # Keep only latest frame
        self.running = False
        self.worker_thread = None
        self.lock = threading.Lock()
        
        # State
        self.emotion_buffer = deque(maxlen=buffer_size)
        self.latest_result = {
            "emotion": "neutral", 
            "confidence": 0.0,
            "is_active": False
        }
        self.frame_count = 0

    def start(self):
        if self.running: return
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def process_frame(self, frame: np.ndarray):
        """Non-blocking push to worker thread."""
        self.frame_count += 1
        if self.frame_count % self.skip_rate != 0:
            return
        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_emotion_data(self) -> Dict:
        """Get smoothed data."""
        with self.lock:
            return self.latest_result

    def _worker_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)  # Shorter timeout
            except queue.Empty:
                continue

            try:
                aligned_face = self._extract_aligned_face(frame)
                if aligned_face is not None:
                    # DeepFace Inference (Fastest settings)
                    results = DeepFace.analyze(
                        img_path=aligned_face,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='skip',
                        silent=True
                    )
                    
                    res = results[0] if isinstance(results, list) else results
                    dominant = res['dominant_emotion']
                    conf = res['emotion'][dominant]

                    with self.lock:
                        self.emotion_buffer.append(dominant)
                        # Calculate mode (most frequent emotion in buffer)
                        if self.emotion_buffer:
                            smoothed = max(set(self.emotion_buffer), key=self.emotion_buffer.count)
                        else:
                            smoothed = dominant
                            
                        self.latest_result = {
                            "emotion": smoothed,
                            "confidence": conf,
                            "is_active": True
                        }
                else:
                    with self.lock:
                         self.latest_result["is_active"] = False
            except Exception:
                # Silently handle any processing errors
                pass

    def _extract_aligned_face(self, frame):
        # MediaPipe Detection -> Alignment -> 48x48 Crop
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections: return None
        
        # Get largest face
        det = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
        kp = det.location_data.relative_keypoints
        
        # Eyes for rotation
        right_eye = (int(kp[0].x * w), int(kp[0].y * h))
        left_eye = (int(kp[1].x * w), int(kp[1].y * h))
        
        # Rotation
        dy = left_eye[1] - right_eye[1]
        dx = left_eye[0] - right_eye[0]
        angle = math.degrees(math.atan2(dy, dx))
        
        bbox = det.location_data.relative_bounding_box
        center = (int((bbox.xmin + bbox.width/2) * w), int((bbox.ymin + bbox.height/2) * h))
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h))
        
        # Crop & Resize
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        
        # Padding 5%
        pad = int(bw * 0.05)
        face = rotated[max(0,y-pad):min(h,y+bh+pad), max(0,x-pad):min(w,x+bw+pad)]
        
        if face.size == 0: return None
        return cv2.resize(face, (48, 48))