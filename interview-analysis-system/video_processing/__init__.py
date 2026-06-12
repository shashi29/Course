"""
Video Processing Module for Interview Analysis System

This module handles all video-related processing including:
- Face detection and recognition
- Facial landmark tracking
- Eye contact and gaze estimation
- Head pose analysis
- Posture analysis
- Facial behavior analysis
"""

# Lazy imports to avoid dependency issues during testing
def __getattr__(name):
    if name == 'FaceDetector':
        from .face_detector import FaceDetector
        return FaceDetector
    elif name == 'FaceLandmarkDetector':
        from .face_landmarks import FaceLandmarkDetector
        return FaceLandmarkDetector
    elif name == 'GazeEstimator':
        from .gaze_estimator import GazeEstimator
        return GazeEstimator
    elif name == 'HeadPoseAnalyzer':
        from .head_pose import HeadPoseAnalyzer
        return HeadPoseAnalyzer
    elif name == 'PostureAnalyzer':
        from .posture_analyzer import PostureAnalyzer
        return PostureAnalyzer
    elif name == 'FacialBehaviorAnalyzer':
        from .facial_behavior import FacialBehaviorAnalyzer
        return FacialBehaviorAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'FaceDetector',
    'FaceLandmarkDetector',
    'GazeEstimator',
    'HeadPoseAnalyzer',
    'PostureAnalyzer',
    'FacialBehaviorAnalyzer'
]
