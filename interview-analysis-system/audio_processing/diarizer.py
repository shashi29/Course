"""
Speaker Diarization using PyAnnote Audio

PyAnnote Audio is the best open-source diarization toolkit,
identifying different speakers in audio recordings.

Link: https://github.com/pyannote/pyannote-audio
"""

import numpy as np
from typing import List, Dict
from pyannote.audio import Pipeline


class SpeakerDiarizer:
    """
    Identifies and separates different speakers in audio using PyAnnote.
    
    Features:
    - Automatic speaker detection
    - Timestamp-based speaker segments
    - Configurable number of speakers
    - High accuracy on conversational audio
    """
    
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1",
                 use_auth_token: str = None):
        """
        Initialize the speaker diarization pipeline.
        
        Args:
            model_name: PyAnnote model name (default: latest version 3.1)
            use_auth_token: HuggingFace authentication token (required for PyAnnote models)
                           Get from: https://huggingface.co/settings/tokens
        """
        self.model_name = model_name
        self.use_auth_token = use_auth_token
        
        print(f"Loading PyAnnote diarization model: {model_name}...")
        
        if use_auth_token:
            self.pipeline = Pipeline.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
        else:
            # Try without token (works for some models)
            try:
                self.pipeline = Pipeline.from_pretrained(model_name)
            except Exception as e:
                print(f"Warning: Authentication may be required. Error: {e}")
                print("Please provide use_auth_token from HuggingFace")
                raise e
        
        print("Diarization pipeline loaded successfully!")
    
    def diarize(self, audio_path: str, min_speakers: int = None, 
                max_speakers: int = None) -> List[Dict]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to the audio file
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)
            
        Returns:
            List of dictionaries containing:
                - start: Start time in seconds
                - end: End time in seconds
                - speaker: Speaker label (e.g., 'SPEAKER_0', 'SPEAKER_1')
        """
        diarization_params = {}
        if min_speakers is not None:
            diarization_params['min_speakers'] = min_speakers
        if max_speakers is not None:
            diarization_params['max_speakers'] = max_speakers
        
        print("Running speaker diarization...")
        diarization = self.pipeline(audio_path, **diarization_params)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        # Get unique speakers
        unique_speakers = list(set([seg['speaker'] for seg in segments]))
        print(f"Detected {len(unique_speakers)} speakers: {unique_speakers}")
        
        return segments
    
    def get_speaker_timeline(self, diarization_result: List[Dict]) -> Dict:
        """
        Create a timeline showing when each speaker spoke.
        
        Args:
            diarization_result: Output from diarize() method
            
        Returns:
            Dictionary mapping speakers to their speaking intervals
        """
        timeline = {}
        
        for segment in diarization_result:
            speaker = segment['speaker']
            if speaker not in timeline:
                timeline[speaker] = []
            timeline[speaker].append({
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['end'] - segment['start']
            })
        
        return timeline
    
    def get_speaker_statistics(self, diarization_result: List[Dict]) -> Dict:
        """
        Calculate statistics about speaker participation.
        
        Args:
            diarization_result: Output from diarize() method
            
        Returns:
            Dictionary with speaker statistics
        """
        stats = {}
        total_duration = 0
        
        # Group by speaker
        speaker_durations = {}
        for segment in diarization_result:
            speaker = segment['speaker']
            duration = segment['end'] - segment['start']
            
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0
            speaker_durations[speaker] += duration
            total_duration += duration
        
        # Calculate percentages
        for speaker, duration in speaker_durations.items():
            stats[speaker] = {
                'total_duration': duration,
                'percentage': (duration / total_duration * 100) if total_duration > 0 else 0,
                'num_segments': sum(1 for s in diarization_result if s['speaker'] == speaker)
            }
        
        stats['total_audio_duration'] = total_duration
        stats['num_speakers'] = len(speaker_durations)
        
        return stats
    
    def detect_number_of_speakers(self, audio_path: str) -> int:
        """
        Automatically detect the number of speakers in audio.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Estimated number of speakers
        """
        segments = self.diarize(audio_path)
        unique_speakers = set([seg['speaker'] for seg in segments])
        return len(unique_speakers)


if __name__ == "__main__":
    # Example usage
    print("Speaker Diarization Module")
    print("=" * 50)
    print("This module uses PyAnnote Audio for speaker identification.")
    print("\nImportant: You need a HuggingFace token to use PyAnnote models.")
    print("Get your token from: https://huggingface.co/settings/tokens")
    print("\nInitialize with:")
    print("  diarizer = SpeakerDiarizer(use_auth_token='YOUR_TOKEN')")
    print("\nDiarize with:")
    print("  segments = diarizer.diarize('audio.wav')")
