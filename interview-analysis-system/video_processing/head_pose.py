"""
Head Pose Analysis using MediaPipe and OpenCV

Analyzes head orientation for reading-screen detection and attention monitoring.

Links:
- MediaPipe: https://google.github.io/mediapipe/
- OpenCV: https://opencv.org/
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class HeadPoseAnalyzer:
    """
    Analyzes head pose (yaw, pitch, roll) from facial landmarks.
    
    Features:
    - Head orientation estimation
    - Reading/screen detection
    - Attention monitoring
    - Movement tracking
    """
    
    # 3D face model points (approximate)
    FACE_MODEL_POINTS = {
        'nose_tip': (0, 0, 0),
        'nose_bottom': (0, -33, 0),
        'left_eye_corner': (-35, 0, -8),
        'right_eye_corner': (35, 0, -8),
        'mouth_left': (-25, -35, -15),
        'mouth_right': (25, -35, -15),
        'chin': (0, -75, 0)
    }
    
    def __init__(self):
        """Initialize the head pose analyzer."""
        print("Initializing Head Pose Analyzer...")
        
        # Camera matrix (will be adjusted based on frame size)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        print("Head Pose Analyzer initialized!")
    
    def estimate_head_pose(self, landmarks: List[List[float]], 
                           frame_shape: Tuple[int, int]) -> Dict:
        """
        Estimate head pose from facial landmarks.
        
        Args:
            landmarks: Facial landmarks (normalized or pixel coordinates)
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Dictionary containing:
                - yaw: Horizontal rotation (degrees)
                - pitch: Vertical rotation (degrees)
                - roll: Tilt rotation (degrees)
                - rotation_vector: Rodrigues rotation vector
                - translation_vector: Translation vector
        """
        h, w = frame_shape[:2]
        
        # Convert landmarks to pixel coordinates if normalized
        if max(landmarks[0]) <= 1.0:
            landmarks_px = [[lm[0] * w, lm[1] * h, lm[2]] for lm in landmarks]
        else:
            landmarks_px = landmarks
        
        # Select key points for pose estimation
        point_3d = np.array([
            self.FACE_MODEL_POINTS['nose_tip'],
            self.FACE_MODEL_POINTS['nose_bottom'],
            self.FACE_MODEL_POINTS['left_eye_corner'],
            self.FACE_MODEL_POINTS['right_eye_corner'],
            self.FACE_MODEL_POINTS['mouth_left'],
            self.FACE_MODEL_POINTS['mouth_right'],
            self.FACE_MODEL_POINTS['chin']
        ], dtype=np.float64)
        
        # Corresponding 2D image points
        point_2d = np.array([
            landmarks_px[1],      # Nose tip
            landmarks_px[6],      # Nose bottom (approximate)
            landmarks_px[33],     # Left eye corner
            landmarks_px[362],    # Right eye corner
            landmarks_px[61],     # Mouth left
            landmarks_px[291],    # Mouth right
            landmarks_px[152]     # Chin
        ], dtype=np.float64)[:, :2]
        
        # Initialize camera matrix
        if self.camera_matrix is None or self.camera_matrix[0, 2] != w/2:
            self.camera_matrix = np.array([
                [w, 0, w/2],
                [0, w, h/2],
                [0, 0, 1]
            ], dtype=np.float64)
        
        # Solve PnP problem
        try:
            success, rot_vec, trans_vec = cv2.solvePnP(
                point_3d, point_2d, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return self._get_default_pose()
            
            # Convert rotation vector to Euler angles
            euler_angles = self._rotation_matrix_to_euler_angles(rot_vec)
            
            return {
                'yaw': float(euler_angles[0]),
                'pitch': float(euler_angles[1]),
                'roll': float(euler_angles[2]),
                'rotation_vector': rot_vec.flatten().tolist(),
                'translation_vector': trans_vec.flatten().tolist()
            }
            
        except Exception as e:
            print(f"Error estimating head pose: {e}")
            return self._get_default_pose()
    
    def _rotation_matrix_to_euler_angles(self, rot_vec: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation vector to Euler angles (yaw, pitch, roll).
        
        Args:
            rot_vec: Rotation vector from solvePnP
            
        Returns:
            Tuple of (yaw, pitch, roll) in degrees
        """
        # Convert to rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rot_vec)
        
        # Extract Euler angles
        sy = np.sqrt(rot_matrix[0, 0]**2 + rot_matrix[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])  # Pitch
            y = np.arctan2(-rot_matrix[2, 0], sy)               # Yaw
            z = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])  # Roll
        else:
            x = np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1])
            y = np.arctan2(-rot_matrix[2, 0], sy)
            z = 0
        
        # Convert to degrees
        return np.degrees(y), np.degrees(x), np.degrees(z)
    
    def _get_default_pose(self) -> Dict:
        """Return default pose values when estimation fails."""
        return {
            'yaw': 0.0,
            'pitch': 0.0,
            'roll': 0.0,
            'rotation_vector': [0, 0, 0],
            'translation_vector': [0, 0, 0]
        }
    
    def detect_reading_behavior(self, pose_results: List[Dict], 
                                 threshold_pitch: float = 20,
                                 threshold_yaw: float = 30) -> Dict:
        """
        Detect reading/screen-looking behavior from head pose.
        
        Args:
            pose_results: List of head pose results over time
            threshold_pitch: Pitch threshold for looking down
            threshold_yaw: Yaw threshold for looking away
            
        Returns:
            Dictionary with reading behavior analysis
        """
        if not pose_results:
            return {'reading_ratio': 0, 'looking_down_ratio': 0}
        
        looking_down_count = 0
        looking_away_count = 0
        
        for pose in pose_results:
            pitch = abs(pose.get('pitch', 0))
            yaw = abs(pose.get('yaw', 0))
            
            if pitch > threshold_pitch:
                looking_down_count += 1
            
            if yaw > threshold_yaw:
                looking_away_count += 1
        
        total = len(pose_results)
        
        return {
            'reading_ratio': float(looking_down_count / total),
            'looking_down_ratio': float(looking_down_count / total),
            'looking_away_ratio': float(looking_away_count / total),
            'attentive_ratio': float((total - looking_down_count - looking_away_count) / total),
            'num_frames_analyzed': total
        }
    
    def calculate_head_movement(self, pose_results: List[Dict]) -> Dict:
        """
        Calculate head movement statistics.
        
        Args:
            pose_results: List of head pose results over time
            
        Returns:
            Dictionary with movement statistics
        """
        if len(pose_results) < 2:
            return {'avg_head_movement': 0, 'max_head_movement': 0}
        
        yaw_changes = []
        pitch_changes = []
        roll_changes = []
        
        for i in range(1, len(pose_results)):
            prev = pose_results[i-1]
            curr = pose_results[i]
            
            yaw_changes.append(abs(curr.get('yaw', 0) - prev.get('yaw', 0)))
            pitch_changes.append(abs(curr.get('pitch', 0) - prev.get('pitch', 0)))
            roll_changes.append(abs(curr.get('roll', 0) - prev.get('roll', 0)))
        
        return {
            'avg_yaw_change': float(np.mean(yaw_changes)),
            'avg_pitch_change': float(np.mean(pitch_changes)),
            'avg_roll_change': float(np.mean(roll_changes)),
            'max_yaw_change': float(max(yaw_changes)),
            'max_pitch_change': float(max(pitch_changes)),
            'max_roll_change': float(max(roll_changes)),
            'total_head_movement': float(np.mean([np.mean(yaw_changes), np.mean(pitch_changes), np.mean(roll_changes)]))
        }
    
    def visualize_pose(self, frame: np.ndarray, pose_result: Dict,
                       landmarks: List[List[float]] = None) -> np.ndarray:
        """
        Visualize head pose on the frame.
        
        Args:
            frame: Original BGR frame
            pose_result: Head pose result dictionary
            landmarks: Optional facial landmarks
            
        Returns:
            Frame with pose visualization
        """
        output_frame = frame.copy()
        h, w, _ = output_frame.shape
        
        # Display angles
        yaw = pose_result.get('yaw', 0)
        pitch = pose_result.get('pitch', 0)
        roll = pose_result.get('roll', 0)
        
        text_y = 30
        cv2.putText(output_frame, f"Yaw: {yaw:.1f}°", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Pitch: {pitch:.1f}°", (10, text_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Roll: {roll:.1f}°", (10, text_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw axis indicator
        center_x, center_y = w - 60, 60
        axis_length = 40
        
        # X-axis (yaw) - red
        end_x = int(center_x + axis_length * np.sin(np.radians(yaw)))
        end_y = int(center_y + axis_length * np.cos(np.radians(yaw)))
        cv2.line(output_frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
        
        return output_frame


if __name__ == "__main__":
    print("Head Pose Analyzer Module")
    print("=" * 50)
    print("This module analyzes head orientation using MediaPipe + OpenCV.")
    print("\nInitialize with:")
    print("  analyzer = HeadPoseAnalyzer()")
    print("\nEstimate pose with:")
    print("  pose = analyzer.estimate_head_pose(landmarks, frame.shape)")
    print("\nDetect reading behavior with:")
    print("  behavior = analyzer.detect_reading_behavior(pose_results)")
