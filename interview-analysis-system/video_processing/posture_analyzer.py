"""
Posture Analysis using MediaPipe Pose

Analyzes sitting posture and body language using MediaPipe Pose
for body landmark tracking.

Link: https://google.github.io/mediapipe/solutions/pose
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import mediapipe as mp


class PostureAnalyzer:
    """
    Analyzes sitting posture and body language from video frames.
    
    Features:
    - Body landmark detection
    - Posture angle calculation
    - Slouching detection
    - Body movement tracking
    """
    
    def __init__(self):
        """Initialize the posture analyzer."""
        print("Loading MediaPipe Pose model...")
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("Pose model loaded successfully!")
    
    def detect_pose(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect body pose landmarks in a frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of dictionaries containing pose information
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return []
        
        h, w, _ = frame.shape
        pose_data = []
        
        for landmark in results.pose_landmarks.landmark:
            pose_data.append({
                'x': landmark.x * w,
                'y': landmark.y * h,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        return [{
            'landmarks': pose_data,
            'num_landmarks': len(pose_data),
            'has_pose': True
        }]
    
    def calculate_posture_angles(self, landmarks: List[Dict]) -> Dict:
        """
        Calculate key posture angles from landmarks.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            Dictionary containing various posture angles
        """
        if len(landmarks) < 33:  # MediaPipe Pose has 33 landmarks
            return self._get_default_angles()
        
        # Key landmark indices
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        
        # Get coordinates
        nose = np.array([landmarks[NOSE]['x'], landmarks[NOSE]['y']])
        left_shoulder = np.array([landmarks[LEFT_SHOULDER]['x'], landmarks[LEFT_SHOULDER]['y']])
        right_shoulder = np.array([landmarks[RIGHT_SHOULDER]['x'], landmarks[RIGHT_SHOULDER]['y']])
        left_hip = np.array([landmarks[LEFT_HIP]['x'], landmarks[LEFT_HIP]['y']])
        right_hip = np.array([landmarks[RIGHT_HIP]['x'], landmarks[RIGHT_HIP]['y']])
        
        # Calculate shoulder angle (forward lean)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        # Vector from hips to shoulders
        torso_vector = shoulder_center - hip_center
        
        # Calculate angle from vertical
        vertical_vector = np.array([0, -1])
        torso_angle = self._angle_between_vectors(torso_vector, vertical_vector)
        
        # Shoulder alignment (hunched vs straight)
        shoulder_line = right_shoulder - left_shoulder
        shoulder_slope = abs(np.arctan2(shoulder_line[1], shoulder_line[0]))
        
        # Head position relative to shoulders
        head_shoulder_vector = nose - shoulder_center
        head_forward = head_shoulder_vector[1]  # Y component indicates forward position
        
        return {
            'torso_angle': float(torso_angle),
            'shoulder_slope': float(np.degrees(shoulder_slope)),
            'head_forward_offset': float(head_forward),
            'is_upright': torso_angle < 20,
            'is_slouching': torso_angle >= 20
        }
    
    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees."""
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        return float(np.degrees(angle))
    
    def _get_default_angles(self) -> Dict:
        """Return default angles when pose detection fails."""
        return {
            'torso_angle': 0.0,
            'shoulder_slope': 0.0,
            'head_forward_offset': 0.0,
            'is_upright': True,
            'is_slouching': False
        }
    
    def analyze_sitting_posture(self, pose_results: List[Dict]) -> Dict:
        """
        Analyze sitting posture over multiple frames.
        
        Args:
            pose_results: List of pose detection results
            
        Returns:
            Dictionary with posture analysis statistics
        """
        if not pose_results:
            return {'avg_torso_angle': 0, 'slouching_ratio': 0}
        
        torso_angles = []
        slouching_count = 0
        
        for result in pose_results:
            if result.get('has_pose') and len(result.get('landmarks', [])) >= 33:
                angles = self.calculate_posture_angles(result['landmarks'])
                torso_angles.append(angles['torso_angle'])
                
                if angles['is_slouching']:
                    slouching_count += 1
        
        total = len(torso_angles)
        
        return {
            'avg_torso_angle': float(np.mean(torso_angles)) if torso_angles else 0,
            'max_torso_angle': float(max(torso_angles)) if torso_angles else 0,
            'min_torso_angle': float(min(torso_angles)) if torso_angles else 0,
            'std_torso_angle': float(np.std(torso_angles)) if len(torso_angles) > 1 else 0,
            'slouching_ratio': float(slouching_count / total) if total > 0 else 0,
            'good_posture_ratio': float((total - slouching_count) / total) if total > 0 else 0,
            'frames_analyzed': total
        }
    
    def visualize_pose(self, frame: np.ndarray, 
                       pose_results: List[Dict]) -> np.ndarray:
        """
        Visualize pose landmarks on the frame.
        
        Args:
            frame: Original BGR frame
            pose_results: Pose detection results
            
        Returns:
            Frame with pose visualization
        """
        output_frame = frame.copy()
        
        for result in pose_results:
            if result.get('has_pose'):
                landmarks = result['landmarks']
                
                # Convert to MediaPipe format
                h, w, _ = frame.shape
                mp_landmarks = []
                for lm in landmarks:
                    mp_landmarks.append(mp.solutions.pose.PoseLandmark(
                        x=lm['x']/w, y=lm['y']/h, z=lm['z'], visibility=lm.get('visibility', 1.0)
                    ))
                
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    mp.solutions.pose.PoseLandmarks(mp_landmarks),
                    self.mp_pose.POSE_CONNECTIONS
                )
        
        return output_frame


if __name__ == "__main__":
    print("Posture Analyzer Module")
    print("=" * 50)
    print("This module uses MediaPipe Pose for posture analysis.")
    print("\nInitialize with:")
    print("  analyzer = PostureAnalyzer()")
    print("\nDetect pose with:")
    print("  pose = analyzer.detect_pose(frame)")
    print("\nCalculate angles with:")
    print("  angles = analyzer.calculate_posture_angles(landmarks)")
