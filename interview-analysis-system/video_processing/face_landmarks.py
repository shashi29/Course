"""
Face Landmark Detection using MediaPipe Face Mesh

MediaPipe Face Mesh provides 468 facial landmarks for detailed
face geometry analysis, enabling eye tracking and expression analysis.

Link: https://google.github.io/mediapipe/solutions/face_mesh
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import mediapipe as mp


class FaceLandmarkDetector:
    """
    Detects 468 facial landmarks using MediaPipe Face Mesh.
    
    Features:
    - 468 precise facial landmarks
    - Eye landmark extraction
    - Mouth landmark extraction
    - Face geometry analysis
    - Real-time performance
    """
    
    # Key landmark indices
    LANDMARK_INDICES = {
        'left_eye': [33, 133, 160, 159, 158, 157, 173],
        'right_eye': [362, 263, 385, 386, 387, 388, 466],
        'left_iris': [474, 475, 476, 477],
        'right_iris': [469, 470, 471, 472],
        'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308],
        'mouth_inner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
        'nose_tip': [1],
        'chin': [152]
    }
    
    def __init__(self, static_image_mode: bool = False, 
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True):
        """
        Initialize the face landmark detector.
        
        Args:
            static_image_mode: Whether to treat input as static images
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Enable iris landmark refinement
        """
        print("Loading MediaPipe Face Mesh model...")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("Face mesh model loaded successfully!")
    
    def detect_landmarks(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect facial landmarks in a video frame.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            List of dictionaries containing landmark information:
                - landmarks: List of 468 (x, y, z) coordinates
                - normalized_landmarks: Normalized coordinates (0-1)
                - eye_landmarks: Extracted eye landmarks
                - mouth_landmarks: Extracted mouth landmarks
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return []
        
        h, w, _ = frame.shape
        face_data = []
        
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            normalized_landmarks = []
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                
                landmarks.append([x, y, z])
                normalized_landmarks.append([landmark.x, landmark.y, landmark.z])
            
            # Extract specific regions
            eye_data = self._extract_eye_landmarks(normalized_landmarks)
            mouth_data = self._extract_mouth_landmarks(normalized_landmarks)
            nose_data = self._extract_nose_landmarks(normalized_landmarks)
            
            face_data.append({
                'landmarks': landmarks,
                'normalized_landmarks': normalized_landmarks,
                'num_landmarks': len(landmarks),
                'left_eye': eye_data['left_eye'],
                'right_eye': eye_data['right_eye'],
                'left_iris': eye_data.get('left_iris'),
                'right_iris': eye_data.get('right_iris'),
                'mouth': mouth_data,
                'nose': nose_data
            })
        
        return face_data
    
    def _extract_eye_landmarks(self, landmarks: List[List[float]]) -> Dict:
        """Extract eye-related landmarks."""
        left_eye = [landmarks[i] for i in self.LANDMARK_INDICES['left_eye']]
        right_eye = [landmarks[i] for i in self.LANDMARK_INDICES['right_eye']]
        
        result = {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'left_eye_center': np.mean(left_eye, axis=0).tolist(),
            'right_eye_center': np.mean(right_eye, axis=0).tolist()
        }
        
        # Add iris landmarks if available
        if len(landmarks) > 477:
            result['left_iris'] = [landmarks[i] for i in self.LANDMARK_INDICES['left_iris']]
            result['right_iris'] = [landmarks[i] for i in self.LANDMARK_INDICES['right_iris']]
        
        return result
    
    def _extract_mouth_landmarks(self, landmarks: List[List[float]]) -> Dict:
        """Extract mouth-related landmarks."""
        mouth_outer = [landmarks[i] for i in self.LANDMARK_INDICES['mouth_outer']]
        mouth_inner = [landmarks[i] for i in self.LANDMARK_INDICES['mouth_inner']]
        
        return {
            'mouth_outer': mouth_outer,
            'mouth_inner': mouth_inner,
            'mouth_center': np.mean(mouth_outer, axis=0).tolist(),
            'mouth_width': np.linalg.norm(np.array(mouth_outer[0][:2]) - np.array(mouth_outer[6][:2])),
            'mouth_height': np.linalg.norm(np.array(mouth_outer[3][:2]) - np.array(mouth_outer[9][:2]))
        }
    
    def _extract_nose_landmarks(self, landmarks: List[List[float]]) -> Dict:
        """Extract nose-related landmarks."""
        nose_tip = landmarks[self.LANDMARK_INDICES['nose_tip'][0]]
        
        return {
            'nose_tip': nose_tip,
            'nose_bridge': landmarks[6],  # Approximate nose bridge
        }
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: List[List[float]]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        Args:
            eye_landmarks: List of eye landmark coordinates
            
        Returns:
            Eye aspect ratio value
        """
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Vertical distances
        v1 = np.linalg.norm(np.array(eye_landmarks[1][:2]) - np.array(eye_landmarks[5][:2]))
        v2 = np.linalg.norm(np.array(eye_landmarks[2][:2]) - np.array(eye_landmarks[4][:2]))
        
        # Horizontal distance
        h = np.linalg.norm(np.array(eye_landmarks[0][:2]) - np.array(eye_landmarks[3][:2]))
        
        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return float(ear)
    
    def detect_blinks(self, landmark_results: List[Dict], 
                      threshold: float = 0.25) -> Dict:
        """
        Detect blinks from facial landmarks.
        
        Args:
            landmark_results: Output from detect_landmarks()
            threshold: EAR threshold for blink detection
            
        Returns:
            Dictionary with blink detection results
        """
        if not landmark_results:
            return {'left_eye_blink': False, 'right_eye_blink': False}
        
        face = landmark_results[0]  # Assume first face
        
        left_ear = self.calculate_eye_aspect_ratio(face['left_eye'])
        right_ear = self.calculate_eye_aspect_ratio(face['right_eye'])
        
        return {
            'left_eye_blink': left_ear < threshold,
            'right_eye_blink': right_ear < threshold,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': (left_ear + right_ear) / 2
        }
    
    def visualize_landmarks(self, frame: np.ndarray, 
                           landmark_results: List[Dict]) -> np.ndarray:
        """
        Visualize facial landmarks on the frame.
        
        Args:
            frame: Original BGR frame
            landmark_results: Output from detect_landmarks()
            
        Returns:
            Frame with landmarks drawn
        """
        output_frame = frame.copy()
        
        for face_landmarks in landmark_results:
            landmarks = face_landmarks['landmarks']
            
            # Convert to MediaPipe format for drawing
            h, w, _ = frame.shape
            mp_landmarks = []
            for lm in landmarks:
                mp_landmarks.append(mp.solutions.face_mesh.FaceLandmark(
                    x=lm[0]/w, y=lm[1]/h, z=lm[2]
                ))
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=mp.solutions.face_mesh.FaceLandmarks(mp_landmarks),
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        return output_frame
    
    def analyze_video(self, video_path: str, 
                      sample_rate: int = 10) -> Dict:
        """
        Analyze facial landmarks throughout a video.
        
        Args:
            video_path: Path to the video file
            sample_rate: Process every nth frame
            
        Returns:
            Dictionary with landmark analysis statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        blink_count = 0
        prev_blink_state = False
        frame_timestamps = []
        ear_values = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                landmarks = self.detect_landmarks(frame)
                
                if landmarks:
                    blink_info = self.detect_blinks(landmarks)
                    ear_values.append(blink_info['avg_ear'])
                    
                    timestamp = frame_idx / fps
                    frame_timestamps.append({
                        'timestamp': timestamp,
                        'ear': blink_info['avg_ear'],
                        'blink': blink_info['left_eye_blink'] or blink_info['right_eye_blink']
                    })
                    
                    # Count blinks (transition from open to closed)
                    current_blink = blink_info['left_eye_blink'] or blink_info['right_eye_blink']
                    if not prev_blink_state and current_blink:
                        blink_count += 1
                    prev_blink_state = current_blink
            
            frame_idx += 1
        
        cap.release()
        
        return {
            'total_frames_processed': frame_idx,
            'sampled_frames': len(frame_timestamps),
            'blink_count': blink_count,
            'blink_rate_per_minute': float(blink_count / (frame_idx / fps) * 60) if frame_idx > 0 else 0,
            'avg_ear': float(np.mean(ear_values)) if ear_values else 0,
            'timeline': frame_timestamps
        }


if __name__ == "__main__":
    # Example usage
    print("Face Landmark Detector Module")
    print("=" * 50)
    print("This module uses MediaPipe Face Mesh for landmark detection.")
    print("\nInitialize with:")
    print("  detector = FaceLandmarkDetector()")
    print("\nDetect landmarks with:")
    print("  landmarks = detector.detect_landmarks(frame)")
    print("\nDetect blinks with:")
    print("  blinks = detector.detect_blinks(landmarks)")
    print("\nAnalyze video with:")
    print("  results = detector.analyze_video('video.mp4')")
