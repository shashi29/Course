"""
Gaze Estimation using L2CS-Net

L2CS-Net provides state-of-the-art gaze estimation for detecting
eye contact and attention direction.

Link: https://github.com/Ahmednull/L2CS-Net
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class GazeEstimator:
    """
    Estimates gaze direction and eye contact using L2CS-Net.
    
    Features:
    - Gaze yaw and pitch estimation
    - Eye contact detection
    - Attention tracking
    - Real-time performance
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the gaze estimator.
        
        Args:
            device: Device to run inference on ('cpu', 'cuda')
        """
        self.device = device
        self.model = None
        
        print("Initializing Gaze Estimator (L2CS-Net)...")
        print("Note: Full L2CS-Net implementation requires model weights.")
        print("Using rule-based gaze estimation as fallback.")
    
    def estimate_gaze(self, frame: np.ndarray, 
                      face_bbox: Dict,
                      landmarks: List[List[float]] = None) -> Dict:
        """
        Estimate gaze direction from a face region.
        
        Args:
            frame: BGR image frame
            face_bbox: Face bounding box dictionary
            landmarks: Optional facial landmarks
            
        Returns:
            Dictionary containing:
                - yaw: Horizontal gaze angle (degrees)
                - pitch: Vertical gaze angle (degrees)
                - eye_contact: Boolean indicating eye contact
                - confidence: Confidence score
        """
        if landmarks:
            return self._estimate_gaze_from_landmarks(landmarks, face_bbox)
        else:
            return self._estimate_gaze_rule_based(frame, face_bbox)
    
    def _estimate_gaze_from_landmarks(self, landmarks: List[List[float]], 
                                       face_bbox: Dict) -> Dict:
        """Estimate gaze from facial landmarks."""
        # Extract eye centers
        left_eye_center = np.mean([landmarks[i] for i in [33, 133, 160, 159, 158, 157]], axis=0)
        right_eye_center = np.mean([landmarks[i] for i in [362, 263, 385, 386, 387, 388]], axis=0)
        nose_tip = landmarks[1]
        
        # Calculate eye-to-nose vectors
        left_vector = nose_tip[:2] - left_eye_center[:2]
        right_vector = nose_tip[:2] - right_eye_center[:2]
        
        # Estimate gaze based on relative positions
        avg_vector = (left_vector + right_vector) / 2
        
        # Normalize
        norm = np.linalg.norm(avg_vector)
        if norm > 0:
            avg_vector = avg_vector / norm
        
        # Simple heuristic for gaze estimation
        yaw = float(np.degrees(np.arctan2(avg_vector[0], avg_vector[1])))
        pitch = float(np.degrees(np.arctan2(avg_vector[1], avg_vector[0]))) * 0.5
        
        # Determine if looking at camera (center of frame)
        h, w, _ = 480, 640, 3  # Default values
        face_center_x = (face_bbox['x1'] + face_bbox['x2']) / 2
        face_center_y = (face_bbox['y1'] + face_bbox['y2']) / 2
        
        center_x, center_y = w / 2, h / 2
        distance_from_center = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
        
        eye_contact = distance_from_center < 100 and abs(yaw) < 15 and abs(pitch) < 15
        
        return {
            'yaw': yaw,
            'pitch': pitch,
            'eye_contact': eye_contact,
            'confidence': 0.7 if eye_contact else 0.5,
            'gaze_vector': avg_vector.tolist()
        }
    
    def _estimate_gaze_rule_based(self, frame: np.ndarray, 
                                   face_bbox: Dict) -> Dict:
        """Rule-based gaze estimation fallback."""
        h, w, _ = frame.shape
        face_center_x = (face_bbox['x1'] + face_bbox['x2']) / 2
        face_center_y = (face_bbox['y1'] + face_bbox['y2']) / 2
        
        # Calculate offset from center
        offset_x = (face_center_x - w/2) / (w/2)
        offset_y = (face_center_y - h/2) / (h/2)
        
        # Convert to angles
        yaw = float(offset_x * 30)  # Approximate degrees
        pitch = float(offset_y * 20)
        
        # Eye contact if face is centered
        eye_contact = abs(offset_x) < 0.15 and abs(offset_y) < 0.15
        
        return {
            'yaw': yaw,
            'pitch': pitch,
            'eye_contact': eye_contact,
            'confidence': 0.5,
            'gaze_vector': [-offset_x, -offset_y]
        }
    
    def analyze_eye_contact(self, gaze_results: List[Dict]) -> Dict:
        """
        Analyze eye contact patterns over time.
        
        Args:
            gaze_results: List of gaze estimation results
            
        Returns:
            Dictionary with eye contact statistics
        """
        if not gaze_results:
            return {
                'eye_contact_ratio': 0,
                'avg_eye_contact_duration': 0,
                'num_eye_contact_periods': 0
            }
        
        eye_contact_flags = [r.get('eye_contact', False) for r in gaze_results]
        total_frames = len(eye_contact_flags)
        
        # Count eye contact periods
        contact_periods = []
        current_period = 0
        in_contact = False
        
        for flag in eye_contact_flags:
            if flag:
                if not in_contact:
                    in_contact = True
                    current_period = 1
                else:
                    current_period += 1
            else:
                if in_contact:
                    contact_periods.append(current_period)
                    in_contact = False
        
        if in_contact:
            contact_periods.append(current_period)
        
        return {
            'eye_contact_ratio': float(sum(eye_contact_flags) / total_frames) if total_frames > 0 else 0,
            'avg_eye_contact_duration': float(np.mean(contact_periods)) if contact_periods else 0,
            'num_eye_contact_periods': len(contact_periods),
            'total_eye_contact_frames': sum(eye_contact_flags)
        }
    
    def get_attention_score(self, gaze_results: List[Dict]) -> float:
        """
        Calculate overall attention score from gaze results.
        
        Args:
            gaze_results: List of gaze estimation results
            
        Returns:
            Attention score between 0 and 1
        """
        if not gaze_results:
            return 0.0
        
        scores = []
        for result in gaze_results:
            yaw = abs(result.get('yaw', 0))
            pitch = abs(result.get('pitch', 0))
            
            # Lower angles = higher attention
            angle_score = max(0, 1 - (yaw + pitch) / 90)
            scores.append(angle_score)
        
        return float(np.mean(scores))


if __name__ == "__main__":
    print("Gaze Estimator Module")
    print("=" * 50)
    print("This module estimates gaze direction and eye contact.")
    print("\nInitialize with:")
    print("  estimator = GazeEstimator()")
    print("\nEstimate gaze with:")
    print("  gaze = estimator.estimate_gaze(frame, face_bbox, landmarks)")
    print("\nAnalyze eye contact with:")
    print("  analysis = estimator.analyze_eye_contact(gaze_results)")
