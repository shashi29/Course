"""
Face Detection using InsightFace

InsightFace provides fast and accurate face analysis capabilities,
including detection, recognition, and attribute analysis.

Link: https://github.com/deepinsight/insightface
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import insightface
from insightface.app import FaceAnalysis


class FaceDetector:
    """
    Detects and analyzes faces in video frames using InsightFace.
    
    Features:
    - Fast face detection
    - Face landmark detection
    - Gender and age estimation
    - Emotion recognition
    - Face recognition/verification
    """
    
    def __init__(self, providers: List[str] = None):
        """
        Initialize the face detector.
        
        Args:
            providers: List of execution providers ('CPUExecutionProvider', 'CUDAExecutionProvider')
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        print("Loading InsightFace model...")
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("Face detector loaded successfully!")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in a video frame.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            List of dictionaries containing face information:
                - bbox: Bounding box coordinates [x1, y1, x2, y2]
                - landmarks: Facial landmark points
                - gender: Estimated gender
                - age: Estimated age
                - embedding: Face embedding vector
                - confidence: Detection confidence
        """
        # Convert BGR to RGB for InsightFace
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.app.get(frame_rgb)
        
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            
            result = {
                'bbox': {
                    'x1': int(bbox[0]),
                    'y1': int(bbox[1]),
                    'x2': int(bbox[2]),
                    'y2': int(bbox[3]),
                    'width': int(bbox[2] - bbox[0]),
                    'height': int(bbox[3] - bbox[1])
                },
                'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                'embedding': face.embedding.tolist() if hasattr(face, 'embedding') else None,
                'gender': 'male' if face.gender == 1 else 'female',
                'age': int(face.age) if hasattr(face, 'age') else None,
                'confidence': float(face.det_score) if hasattr(face, 'det_score') else None
            }
            
            results.append(result)
        
        return results
    
    def detect_faces_from_image_path(self, image_path: str) -> List[Dict]:
        """
        Detect faces from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Same format as detect_faces() method
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return self.detect_faces(frame)
    
    def count_faces(self, frame: np.ndarray) -> int:
        """
        Count number of faces in a frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Number of detected faces
        """
        return len(self.detect_faces(frame))
    
    def get_face_crop(self, frame: np.ndarray, face_result: Dict, 
                      padding: float = 0.2) -> np.ndarray:
        """
        Crop a face region from the frame.
        
        Args:
            frame: Original frame
            face_result: Face detection result dictionary
            padding: Padding around face (fraction of face size)
            
        Returns:
            Cropped face image
        """
        bbox = face_result['bbox']
        h, w = frame.shape[:2]
        
        # Calculate padded coordinates
        face_w = bbox['x2'] - bbox['x1']
        face_h = bbox['y2'] - bbox['y1']
        
        pad_x = int(face_w * padding)
        pad_y = int(face_h * padding)
        
        x1 = max(0, bbox['x1'] - pad_x)
        y1 = max(0, bbox['y1'] - pad_y)
        x2 = min(w, bbox['x2'] + pad_x)
        y2 = min(h, bbox['y2'] + pad_y)
        
        return frame[y1:y2, x1:x2]
    
    def verify_faces(self, face_embedding_1: np.ndarray, 
                     face_embedding_2: np.ndarray,
                     threshold: float = 0.5) -> Dict:
        """
        Verify if two face embeddings belong to the same person.
        
        Args:
            face_embedding_1: First face embedding
            face_embedding_2: Second face embedding
            threshold: Similarity threshold
            
        Returns:
            Dictionary with verification result
        """
        # Normalize embeddings
        emb1 = face_embedding_1 / np.linalg.norm(face_embedding_1)
        emb2 = face_embedding_2 / np.linalg.norm(face_embedding_2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2)
        
        return {
            'is_same_person': similarity >= threshold,
            'similarity_score': float(similarity),
            'threshold': threshold
        }
    
    def track_faces(self, previous_faces: List[Dict], 
                    current_frame: np.ndarray,
                    iou_threshold: float = 0.5) -> List[Dict]:
        """
        Simple face tracking based on IoU (Intersection over Union).
        
        Args:
            previous_faces: Face detections from previous frame
            current_frame: Current video frame
            iou_threshold: IoU threshold for matching
            
        Returns:
            List of tracked faces with IDs
        """
        current_faces = self.detect_faces(current_frame)
        
        # Assign IDs to new faces
        for i, face in enumerate(current_faces):
            if i < len(previous_faces):
                face['track_id'] = previous_faces[i].get('track_id', i)
            else:
                face['track_id'] = len(previous_faces) + i
        
        return current_faces
    
    def analyze_video(self, video_path: str, 
                      sample_rate: int = 30) -> Dict:
        """
        Analyze faces throughout a video.
        
        Args:
            video_path: Path to the video file
            sample_rate: Process every nth frame
            
        Returns:
            Dictionary with face analysis statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        face_counts = []
        unique_faces = []
        frame_timestamps = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                faces = self.detect_faces(frame)
                face_counts.append(len(faces))
                
                timestamp = frame_idx / fps
                frame_timestamps.append(timestamp)
                
                # Store first occurrence of each face
                for face in faces:
                    if face['embedding'] is not None:
                        unique_faces.append({
                            'frame': frame_idx,
                            'timestamp': timestamp,
                            'embedding': face['embedding'],
                            'bbox': face['bbox']
                        })
            
            frame_idx += 1
        
        cap.release()
        
        return {
            'total_frames_processed': frame_idx,
            'sampled_frames': len(face_counts),
            'avg_faces_per_frame': float(np.mean(face_counts)) if face_counts else 0,
            'max_faces_detected': int(max(face_counts)) if face_counts else 0,
            'video_duration_seconds': float(frame_idx / fps),
            'face_detections': unique_faces
        }


if __name__ == "__main__":
    # Example usage
    print("Face Detector Module")
    print("=" * 50)
    print("This module uses InsightFace for face detection.")
    print("\nInitialize with:")
    print("  detector = FaceDetector()")
    print("\nDetect faces with:")
    print("  faces = detector.detect_faces(frame)")
    print("\nAnalyze video with:")
    print("  results = detector.analyze_video('video.mp4')")
