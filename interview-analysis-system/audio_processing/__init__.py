"""
Audio Processing Module for Interview Analysis System

This module handles all audio-related processing including:
- Speech-to-Text transcription
- Speaker diarization
- Voice embeddings and verification
- Audio feature extraction
- Emotion detection from speech
"""

# Lazy imports to avoid dependency issues during testing
def __getattr__(name):
    if name == 'SpeechToTextTranscriber':
        from .transcriber import SpeechToTextTranscriber
        return SpeechToTextTranscriber
    elif name == 'SpeakerDiarizer':
        from .diarizer import SpeakerDiarizer
        return SpeakerDiarizer
    elif name == 'VoiceEmbeddingExtractor':
        from .voice_embeddings import VoiceEmbeddingExtractor
        return VoiceEmbeddingExtractor
    elif name == 'AudioFeatureExtractor':
        from .audio_features import AudioFeatureExtractor
        return AudioFeatureExtractor
    elif name == 'EmotionDetector':
        from .emotion_detector import EmotionDetector
        return EmotionDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'SpeechToTextTranscriber',
    'SpeakerDiarizer',
    'VoiceEmbeddingExtractor',
    'AudioFeatureExtractor',
    'EmotionDetector'
]
