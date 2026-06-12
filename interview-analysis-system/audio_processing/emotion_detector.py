"""
Emotion Detection from Speech using SpeechBrain

SpeechBrain provides open-source emotion classification models
for detecting engagement, stress, confidence, and other emotional states.

Link: https://github.com/speechbrain/speechbrain
"""

import numpy as np
from typing import Dict, List, Optional
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier


class EmotionDetector:
    """
    Detects emotions from speech including engagement, stress, and confidence.
    
    Features:
    - Multi-class emotion classification
    - Emotion probability distributions
    - Temporal emotion tracking
    - Confidence scores
    """
    
    # Common emotion categories
    EMOTION_CATEGORIES = [
        'neutral',
        'happy',
        'sad',
        'angry',
        'fearful',
        'disgusted',
        'surprised',
        'excited',
        'stressed',
        'confident'
    ]
    
    def __init__(self, model_name: str = "speechbrain/emotion-recognition-wav2vec2-MSP",
                 device: str = "cpu"):
        """
        Initialize the emotion detection model.
        
        Args:
            model_name: SpeechBrain model name for emotion recognition
            device: Device to run inference on ('cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading SpeechBrain emotion model: {model_name}...")
        try:
            self.classifier = EncoderClassifier.from_hparams(
                source=model_name,
                savedir=f"/tmp/speechbrain/{model_name.split('/')[-1]}",
                run_opts={"device": device}
            )
            print("Emotion model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load emotion model. Error: {e}")
            print("Emotion detection will use rule-based heuristics as fallback.")
            self.classifier = None
    
    def detect_emotion(self, audio_path: str) -> Dict:
        """
        Detect emotion from audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing:
                - predicted_emotion: Most likely emotion
                - probabilities: Probability distribution over emotions
                - confidence: Confidence score of prediction
        """
        if self.classifier is None:
            return self._rule_based_emotion_detection(audio_path)
        
        try:
            # Get emotion prediction
            prediction = self.classifier.classify_file(audio_path)
            
            # Parse prediction (format depends on model)
            if isinstance(prediction, list) and len(prediction) > 0:
                emotion_probs = prediction[0]
            else:
                emotion_probs = prediction
            
            # Find most likely emotion
            if isinstance(emotion_probs, dict):
                predicted_emotion = max(emotion_probs, key=emotion_probs.get)
                confidence = emotion_probs[predicted_emotion]
            else:
                # Handle array/tensor output
                probs_array = emotion_probs.cpu().numpy() if hasattr(emotion_probs, 'cpu') else emotion_probs
                predicted_idx = np.argmax(probs_array)
                predicted_emotion = self.EMOTION_CATEGORIES[predicted_idx % len(self.EMOTION_CATEGORIES)]
                confidence = float(np.max(probs_array))
                emotion_probs = {self.EMOTION_CATEGORIES[i]: float(probs_array[i]) 
                                for i in range(min(len(probs_array), len(self.EMOTION_CATEGORIES)))}
            
            return {
                'predicted_emotion': predicted_emotion,
                'probabilities': emotion_probs,
                'confidence': float(confidence) if not isinstance(confidence, dict) else 0.5
            }
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return self._rule_based_emotion_detection(audio_path)
    
    def detect_emotion_from_array(self, audio_array: np.ndarray, 
                                   sample_rate: int = 16000) -> Dict:
        """
        Detect emotion from audio array.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Same format as detect_emotion() method
        """
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_array, sample_rate)
            result = self.detect_emotion(tmp_file.name)
        
        return result
    
    def analyze_emotion_timeline(self, audio_path: str, 
                                  segment_duration: float = 2.0) -> List[Dict]:
        """
        Analyze emotion changes over time in audio.
        
        Args:
            audio_path: Path to the audio file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of emotion predictions for each time segment
        """
        import soundfile as sf
        
        # Load audio
        audio_array, sample_rate = sf.read(audio_path)
        total_duration = len(audio_array) / sample_rate
        
        timeline = []
        current_time = 0
        
        while current_time < total_duration:
            start_sample = int(current_time * sample_rate)
            end_sample = int(min((current_time + segment_duration) * sample_rate, len(audio_array)))
            
            segment = audio_array[start_sample:end_sample]
            
            # Skip very short segments
            if len(segment) / sample_rate < 0.5:
                current_time += segment_duration
                continue
            
            emotion_result = self.detect_emotion_from_array(segment, sample_rate)
            
            timeline.append({
                'start_time': current_time,
                'end_time': current_time + segment_duration,
                **emotion_result
            })
            
            current_time += segment_duration
        
        return timeline
    
    def calculate_engagement_score(self, emotion_timeline: List[Dict]) -> Dict:
        """
        Calculate overall engagement score from emotion timeline.
        
        Args:
            emotion_timeline: Output from analyze_emotion_timeline()
            
        Returns:
            Dictionary containing engagement metrics
        """
        positive_emotions = ['happy', 'excited', 'confident', 'surprised']
        negative_emotions = ['sad', 'angry', 'fearful', 'disgusted', 'stressed']
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        confidence_scores = []
        
        for segment in emotion_timeline:
            emotion = segment.get('predicted_emotion', 'neutral').lower()
            confidence = segment.get('confidence', 0.5)
            confidence_scores.append(confidence)
            
            if emotion in positive_emotions:
                positive_count += 1
            elif emotion in negative_emotions:
                negative_count += 1
            else:
                neutral_count += 1
        
        total = len(emotion_timeline)
        if total == 0:
            return {
                'engagement_score': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'neutral_ratio': 0,
                'avg_confidence': 0
            }
        
        # Engagement score: higher positive ratio and lower negative ratio = more engaged
        positive_ratio = positive_count / total
        negative_ratio = negative_count / total
        engagement_score = (positive_ratio - negative_ratio + 1) / 2  # Normalize to 0-1
        
        return {
            'engagement_score': float(engagement_score),
            'positive_ratio': float(positive_ratio),
            'negative_ratio': float(negative_ratio),
            'neutral_ratio': float(neutral_count / total),
            'avg_confidence': float(np.mean(confidence_scores)),
            'total_segments': total
        }
    
    def detect_stress_indicators(self, audio_features: Dict) -> Dict:
        """
        Detect stress indicators from audio features (rule-based fallback).
        
        Args:
            audio_features: Dictionary of audio features from AudioFeatureExtractor
            
        Returns:
            Dictionary containing stress indicators
        """
        stress_score = 0
        indicators = []
        
        # Check pitch variability (high variability can indicate stress)
        pitch_info = audio_features.get('pitch', {})
        if pitch_info.get('std_pitch', 0) > 50:
            stress_score += 0.3
            indicators.append("High pitch variability")
        
        # Check speaking rate (very fast or very slow can indicate stress)
        wpm_info = audio_features.get('wpm', {})
        wpm = wpm_info.get('wpm', 150)
        if wpm > 180 or wpm < 100:
            stress_score += 0.2
            indicators.append(f"Unusual speaking rate: {wpm:.1f} WPM")
        
        # Check silence ratio (too much silence can indicate hesitation/stress)
        silence_info = audio_features.get('silence_pauses', {})
        silence_ratio = silence_info.get('silence_ratio', 0)
        if silence_ratio > 0.3:
            stress_score += 0.2
            indicators.append("High silence ratio")
        
        # Check energy level (low energy can indicate stress/fatigue)
        energy_info = audio_features.get('energy', {})
        if energy_info.get('mean_energy', 0.1) < 0.05:
            stress_score += 0.3
            indicators.append("Low energy level")
        
        return {
            'stress_score': min(float(stress_score), 1.0),
            'stress_level': 'high' if stress_score > 0.6 else 'medium' if stress_score > 0.3 else 'low',
            'indicators': indicators
        }
    
    def _rule_based_emotion_detection(self, audio_path: str) -> Dict:
        """
        Fallback rule-based emotion detection when model is unavailable.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with estimated emotion based on audio features
        """
        # Use audio features to make educated guess
        extractor = AudioFeatureExtractor()
        features = extractor.extract_all_features(audio_path)
        
        # Simple heuristic-based emotion estimation
        pitch_std = features['pitch'].get('std_pitch', 0)
        energy_mean = features['energy'].get('mean_energy', 0)
        
        if pitch_std > 40 and energy_mean > 0.1:
            emotion = 'excited'
            confidence = 0.6
        elif pitch_std < 20 and energy_mean < 0.05:
            emotion = 'sad'
            confidence = 0.5
        elif pitch_std > 30:
            emotion = 'stressed'
            confidence = 0.55
        else:
            emotion = 'neutral'
            confidence = 0.5
        
        return {
            'predicted_emotion': emotion,
            'probabilities': {emotion: confidence},
            'confidence': confidence,
            'method': 'rule_based'
        }


# Import for fallback
from .audio_features import AudioFeatureExtractor


if __name__ == "__main__":
    # Example usage
    print("Emotion Detector Module")
    print("=" * 50)
    print("This module uses SpeechBrain for emotion detection.")
    print("\nInitialize with:")
    print("  detector = EmotionDetector()")
    print("\nDetect emotion with:")
    print("  result = detector.detect_emotion('audio.wav')")
    print("\nAnalyze emotion timeline with:")
    print("  timeline = detector.analyze_emotion_timeline('audio.wav')")
    print("\nCalculate engagement with:")
    print("  engagement = detector.calculate_engagement_score(timeline)")
