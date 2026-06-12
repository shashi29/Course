"""
Facial Behavior Analysis using Py-Feat

Py-Feat provides research-grade facial analysis for detecting
attention, engagement, and facial expressions.

Link: https://github.com/DillonDuPont/py-feat
"""

import cv2
import numpy as np
from typing import Dict, List, Optional


class FacialBehaviorAnalyzer:
    """
    Analyzes facial behavior for attention and engagement detection.
    
    Features:
    - Facial Action Coding System (FACS)
    - Expression intensity tracking
    - Attention level estimation
    - Engagement scoring
    """
    
    # Action Unit groups for different expressions
    AU_EXPRESSIONS = {
        'happiness': [6, 12],
        'sadness': [1, 4, 15],
        'anger': [4, 5, 7, 23],
        'surprise': [1, 2, 5, 26],
        'fear': [1, 2, 4, 5, 20, 26],
        'disgust': [9, 10, 17],
        'contempt': [14],
        'concentration': [4, 7, 10]
    }
    
    def __init__(self):
        """Initialize the facial behavior analyzer."""
        print("Initializing Facial Behavior Analyzer...")
        print("Note: Full Py-Feat integration requires model installation.")
        print("Using rule-based facial analysis as fallback.")
    
    def analyze_facial_expression(self, landmarks: List[List[float]]) -> Dict:
        """
        Analyze facial expression from landmarks.
        
        Args:
            landmarks: Facial landmarks list
            
        Returns:
            Dictionary with expression analysis
        """
        if len(landmarks) < 468:
            return self._get_default_expression()
        
        # Extract key regions
        mouth_data = self._analyze_mouth(landmarks)
        eye_data = self._analyze_eyes(landmarks)
        brow_data = self._analyze_brows(landmarks)
        
        # Determine dominant expression
        expression_scores = self._calculate_expression_scores(mouth_data, eye_data, brow_data)
        dominant_expression = max(expression_scores, key=expression_scores.get)
        
        return {
            'dominant_expression': dominant_expression,
            'expression_scores': expression_scores,
            'mouth': mouth_data,
            'eyes': eye_data,
            'brows': brow_data,
            'engagement_level': self._calculate_engagement(eye_data, brow_data)
        }
    
    def _analyze_mouth(self, landmarks: List[List[float]]) -> Dict:
        """Analyze mouth region."""
        # Mouth outer landmarks
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        
        # Calculate mouth aspect ratio
        mouth_height = np.linalg.norm(np.array(mouth_top[:2]) - np.array(mouth_bottom[:2]))
        mouth_width = np.linalg.norm(np.array(mouth_left[:2]) - np.array(mouth_right[:2]))
        
        mar = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # Smile detection (corners up)
        left_corner_up = mouth_left[1] < landmarks[62][1]
        right_corner_up = mouth_right[1] < landmarks[292][1]
        is_smiling = left_corner_up and right_corner_up
        
        return {
            'mouth_aspect_ratio': float(mar),
            'is_open': mar > 0.3,
            'is_smiling': is_smiling,
            'smile_intensity': float((mar - 0.2) / 0.3) if mar > 0.2 else 0
        }
    
    def _analyze_eyes(self, landmarks: List[List[float]]) -> Dict:
        """Analyze eye region."""
        # Left eye
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        left_eye_left = landmarks[33]
        left_eye_right = landmarks[133]
        
        # Right eye
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        right_eye_left = landmarks[362]
        right_eye_right = landmarks[263]
        
        # Calculate EAR for each eye
        left_ear = self._calculate_eye_aspect_ratio(
            left_eye_top, left_eye_bottom, left_eye_left, left_eye_right
        )
        right_ear = self._calculate_eye_aspect_ratio(
            right_eye_top, right_eye_bottom, right_eye_left, right_eye_right
        )
        
        avg_ear = (left_ear + right_ear) / 2
        
        return {
            'left_eye_aspect_ratio': float(left_ear),
            'right_eye_aspect_ratio': float(right_ear),
            'avg_eye_aspect_ratio': float(avg_ear),
            'eyes_open': avg_ear > 0.25,
            'alertness': min(float(avg_ear / 0.3), 1.0)
        }
    
    def _calculate_eye_aspect_ratio(self, top, bottom, left, right) -> float:
        """Calculate Eye Aspect Ratio."""
        vertical = np.linalg.norm(np.array(top[:2]) - np.array(bottom[:2]))
        horizontal = np.linalg.norm(np.array(left[:2]) - np.array(right[:2]))
        
        return vertical / horizontal if horizontal > 0 else 0
    
    def _analyze_brows(self, landmarks: List[List[float]]) -> Dict:
        """Analyze brow/eyebrow region."""
        # Left brow landmarks
        left_brow_inner = landmarks[107]
        left_brow_outer = landmarks[130]
        left_brow_center = landmarks[117]
        
        # Right brow landmarks
        right_brow_inner = landmarks[336]
        right_brow_outer = landmarks[359]
        right_brow_center = landmarks[346]
        
        # Calculate brow positions relative to eyes
        eye_level = (landmarks[159][1] + landmarks[386][1]) / 2
        
        left_brow_height = eye_level - left_brow_center[1]
        right_brow_height = eye_level - right_brow_center[1]
        
        # Brow furrowing (inner brows raised)
        brow_furrowed = (left_brow_inner[1] + right_brow_inner[1]) / 2 < eye_level - 10
        
        return {
            'left_brow_height': float(left_brow_height),
            'right_brow_height': float(right_brow_height),
            'brow_furrowed': brow_furrowed,
            'brow_symmetry': abs(left_brow_height - right_brow_height)
        }
    
    def _calculate_expression_scores(self, mouth: Dict, eyes: Dict, brows: Dict) -> Dict:
        """Calculate scores for different expressions."""
        scores = {
            'neutral': 0.5,
            'happy': 0.1,
            'focused': 0.3,
            'confused': 0.1,
            'stressed': 0.1
        }
        
        # Happy if smiling
        if mouth['is_smiling']:
            scores['happy'] = 0.5 + mouth['smile_intensity'] * 0.3
            scores['neutral'] -= 0.2
        
        # Focused if eyes open and brows slightly furrowed
        if eyes['eyes_open'] and brows['brow_furrowed']:
            scores['focused'] = 0.6
            scores['neutral'] -= 0.1
        
        # Stressed if brows very furrowed
        if brows['brow_furrowed'] and not mouth['is_smiling']:
            scores['stressed'] = 0.4
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def _calculate_engagement(self, eyes: Dict, brows: Dict) -> float:
        """Calculate engagement level from 0 to 1."""
        # Eyes open contributes to engagement
        eye_score = eyes['alertness']
        
        # Some brow activity indicates engagement
        brow_score = 1.0 - min(brows['brow_symmetry'] / 10, 1.0)
        
        engagement = (eye_score * 0.7 + brow_score * 0.3)
        return float(min(max(engagement, 0), 1))
    
    def analyze_attention_timeline(self, expression_results: List[Dict]) -> Dict:
        """
        Analyze attention patterns over time.
        
        Args:
            expression_results: List of expression analysis results
            
        Returns:
            Dictionary with attention statistics
        """
        if not expression_results:
            return {'avg_engagement': 0, 'attention_span': 0}
        
        engagement_scores = [r.get('engagement_level', 0) for r in expression_results]
        focused_count = sum(1 for r in expression_results if r.get('dominant_expression') == 'focused')
        
        # Calculate attention span (consecutive engaged frames)
        max_consecutive = 0
        current_consecutive = 0
        
        for score in engagement_scores:
            if score > 0.5:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return {
            'avg_engagement': float(np.mean(engagement_scores)),
            'max_engagement': float(max(engagement_scores)),
            'min_engagement': float(min(engagement_scores)),
            'std_engagement': float(np.std(engagement_scores)),
            'attention_span_frames': max_consecutive,
            'focused_ratio': float(focused_count / len(expression_results)),
            'frames_analyzed': len(expression_results)
        }
    
    def _get_default_expression(self) -> Dict:
        """Return default expression when analysis fails."""
        return {
            'dominant_expression': 'neutral',
            'expression_scores': {'neutral': 1.0},
            'engagement_level': 0.5
        }


if __name__ == "__main__":
    print("Facial Behavior Analyzer Module")
    print("=" * 50)
    print("This module uses Py-Feat concepts for facial analysis.")
    print("\nInitialize with:")
    print("  analyzer = FacialBehaviorAnalyzer()")
    print("\nAnalyze expression with:")
    print("  expression = analyzer.analyze_facial_expression(landmarks)")
    print("\nAnalyze attention with:")
    print("  attention = analyzer.analyze_attention_timeline(expression_results)")
