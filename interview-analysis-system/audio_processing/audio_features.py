"""
Audio Feature Extraction using OpenSMILE and Librosa

OpenSMILE is the industry standard for audio feature extraction,
while Librosa provides comprehensive audio analytics capabilities.

Links:
- OpenSMILE: https://github.com/audeering/opensmile-python
- Librosa: https://librosa.org/
"""

import numpy as np
from typing import Dict, List, Optional
import librosa
import librosa.feature


class AudioFeatureExtractor:
    """
    Extracts various audio features including pitch, tone, energy, 
    silence detection, pauses, and words per minute (WPM).
    
    Features:
    - Pitch (F0) extraction
    - Energy and intensity analysis
    - Spectral features (MFCCs, chroma, spectral contrast)
    - Silence and pause detection
    - Speaking rate estimation (WPM)
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio feature extractor.
        
        Args:
            sample_rate: Sample rate for audio processing (default: 16000 Hz)
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path: str) -> tuple:
        """
        Load audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        return y, sr
    
    def extract_pitch(self, audio_array: np.ndarray, 
                      sample_rate: int = None) -> Dict:
        """
        Extract pitch (fundamental frequency) features.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary containing pitch statistics
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Extract fundamental frequency (F0) using librosa's pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_array,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate
        )
        
        # Handle NaN values (unvoiced regions)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            return {
                'mean_pitch': 0,
                'std_pitch': 0,
                'min_pitch': 0,
                'max_pitch': 0,
                'pitch_range': 0,
                'voiced_ratio': 0
            }
        
        return {
            'mean_pitch': float(np.mean(f0_clean)),
            'std_pitch': float(np.std(f0_clean)),
            'min_pitch': float(np.min(f0_clean)),
            'max_pitch': float(np.max(f0_clean)),
            'pitch_range': float(np.max(f0_clean) - np.min(f0_clean)),
            'voiced_ratio': float(np.sum(voiced_flag) / len(voiced_flag)) if len(voiced_flag) > 0 else 0,
            'f0_contour': f0_clean.tolist()
        }
    
    def extract_energy(self, audio_array: np.ndarray) -> Dict:
        """
        Extract energy and intensity features.
        
        Args:
            audio_array: Audio data as numpy array
            
        Returns:
            Dictionary containing energy statistics
        """
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio_array)[0]
        
        # Calculate zero-crossing rate (related to energy)
        zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
        
        return {
            'mean_energy': float(np.mean(rms)),
            'std_energy': float(np.std(rms)),
            'min_energy': float(np.min(rms)),
            'max_energy': float(np.max(rms)),
            'energy_variance': float(np.var(rms)),
            'mean_zero_crossing_rate': float(np.mean(zcr)),
            'rms_contour': rms.tolist()
        }
    
    def extract_spectral_features(self, audio_array: np.ndarray, 
                                  sample_rate: int = None) -> Dict:
        """
        Extract spectral features (MFCCs, chroma, spectral contrast).
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary containing spectral features
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_array, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio_array, sr=sample_rate)
        contrast_mean = np.mean(contrast.T, axis=0)
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)[0]
        
        return {
            'mfccs_mean': mfccs_mean.tolist(),
            'mfccs_std': mfccs_std.tolist(),
            'chroma_mean': chroma_mean.tolist(),
            'spectral_contrast': contrast_mean.tolist(),
            'mean_spectral_centroid': float(np.mean(centroid)),
            'spectral_rolloff': float(librosa.feature.spectral_rolloff(y=audio_array, sr=sample_rate)[0].mean())
        }
    
    def detect_silence_and_pauses(self, audio_array: np.ndarray, 
                                   sample_rate: int = None,
                                   threshold_db: float = -40,
                                   min_silence_duration: float = 0.3) -> Dict:
        """
        Detect silence and pauses in audio.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            threshold_db: Threshold in dB for silence detection
            min_silence_duration: Minimum duration (seconds) to consider as silence
            
        Returns:
            Dictionary containing silence/pause statistics
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Convert to dB
        audio_db = librosa.amplitude_to_db(np.abs(audio_array), ref=np.max)
        
        # Find silent regions
        silence_mask = audio_db < threshold_db
        
        # Find transitions
        diff = np.diff(silence_mask.astype(int))
        silence_starts = np.where(diff == 1)[0] + 1
        silence_ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if silence_mask[0]:
            silence_starts = np.insert(silence_starts, 0, 0)
        if silence_mask[-1]:
            silence_ends = np.append(silence_ends, len(audio_db))
        
        # Calculate silence durations
        silence_durations = []
        for start, end in zip(silence_starts, silence_ends):
            duration = (end - start) / sample_rate
            if duration >= min_silence_duration:
                silence_durations.append({
                    'start': float(start / sample_rate),
                    'end': float(end / sample_rate),
                    'duration': float(duration)
                })
        
        total_silence = sum([s['duration'] for s in silence_durations])
        total_duration = len(audio_array) / sample_rate
        
        return {
            'total_silence_duration': float(total_silence),
            'silence_ratio': float(total_silence / total_duration) if total_duration > 0 else 0,
            'num_silent_segments': len(silence_durations),
            'avg_silence_duration': float(total_silence / len(silence_durations)) if silence_durations else 0,
            'silent_segments': silence_durations
        }
    
    def estimate_wpm(self, transcription_segments: List[Dict], 
                     audio_duration: float) -> Dict:
        """
        Estimate words per minute from transcription segments.
        
        Args:
            transcription_segments: List of transcription segments with text
            audio_duration: Total audio duration in seconds
            
        Returns:
            Dictionary containing WPM statistics
        """
        total_words = 0
        word_counts = []
        
        for segment in transcription_segments:
            text = segment.get('text', '')
            words = text.strip().split()
            word_count = len(words)
            total_words += word_count
            word_counts.append(word_count)
        
        # Calculate WPM
        duration_minutes = audio_duration / 60
        wpm = total_words / duration_minutes if duration_minutes > 0 else 0
        
        return {
            'total_words': total_words,
            'wpm': float(wpm),
            'avg_words_per_segment': float(np.mean(word_counts)) if word_counts else 0,
            'audio_duration_seconds': float(audio_duration)
        }
    
    def extract_all_features(self, audio_path: str, 
                             transcription_segments: List[Dict] = None) -> Dict:
        """
        Extract all audio features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            transcription_segments: Optional transcription segments for WPM calculation
            
        Returns:
            Comprehensive dictionary of all audio features
        """
        # Load audio
        audio_array, sample_rate = self.load_audio(audio_path)
        
        # Extract all features
        features = {
            'pitch': self.extract_pitch(audio_array, sample_rate),
            'energy': self.extract_energy(audio_array),
            'spectral': self.extract_spectral_features(audio_array, sample_rate),
            'silence_pauses': self.detect_silence_and_pauses(audio_array, sample_rate)
        }
        
        # Add WPM if transcription provided
        if transcription_segments:
            audio_duration = len(audio_array) / sample_rate
            features['wpm'] = self.estimate_wpm(transcription_segments, audio_duration)
        
        # Add basic audio info
        features['audio_info'] = {
            'duration_seconds': float(len(audio_array) / sample_rate),
            'sample_rate': sample_rate,
            'num_samples': len(audio_array)
        }
        
        return features


if __name__ == "__main__":
    # Example usage
    print("Audio Feature Extractor Module")
    print("=" * 50)
    print("This module uses Librosa for audio feature extraction.")
    print("\nInitialize with:")
    print("  extractor = AudioFeatureExtractor()")
    print("\nExtract all features with:")
    print("  features = extractor.extract_all_features('audio.wav')")
    print("\nIndividual feature extraction methods:")
    print("  - extract_pitch(audio_array)")
    print("  - extract_energy(audio_array)")
    print("  - extract_spectral_features(audio_array)")
    print("  - detect_silence_and_pauses(audio_array)")
    print("  - estimate_wpm(transcription_segments, audio_duration)")
