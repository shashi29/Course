"""
Speech-to-Text Transcriber using Faster-Whisper

Faster-Whisper is the fastest production Whisper implementation,
providing efficient and accurate speech transcription.

Link: https://github.com/guillaumekln/faster-whisper
"""

import numpy as np
from typing import List, Dict, Optional
from faster_whisper import WhisperModel


class SpeechToTextTranscriber:
    """
    Transcribes audio to text using Faster-Whisper model.
    
    Features:
    - Multiple model sizes (tiny, base, small, medium, large)
    - GPU acceleration support
    - Word-level timestamps
    - Multiple language support
    """
    
    def __init__(self, model_size: str = "base", device: str = "auto", 
                 compute_type: str = "float32"):
        """
        Initialize the transcriber with specified model.
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            compute_type: Computation precision ('float32', 'float16', 'int8')
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        print(f"Loading Faster-Whisper model: {model_size}...")
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type
        )
        print("Model loaded successfully!")
    
    def transcribe(self, audio_path: str, language: str = "en", 
                   word_timestamps: bool = True) -> Dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            language: Language code for transcription (default: 'en')
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Dictionary containing:
                - transcript: Full transcription text
                - segments: List of segments with timestamps
                - language: Detected language
        """
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps
        )
        
        segments_list = []
        full_transcript = []
        
        for segment in segments:
            segment_dict = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'confidence': segment.avg_logprob
            }
            
            if word_timestamps and hasattr(segment, 'words'):
                words = []
                for word in segment.words:
                    words.append({
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'confidence': word.probability
                    })
                segment_dict['words'] = words
            
            segments_list.append(segment_dict)
            full_transcript.append(segment.text)
        
        return {
            'transcript': ''.join(full_transcript),
            'segments': segments_list,
            'language': info.language,
            'language_probability': info.language_probability
        }
    
    def transcribe_from_array(self, audio_array: np.ndarray, 
                              sample_rate: int = 16000,
                              language: str = "en") -> Dict:
        """
        Transcribe audio from numpy array.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            language: Language code for transcription
            
        Returns:
            Same format as transcribe() method
        """
        # Save temporarily and transcribe
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_array, sample_rate)
            result = self.transcribe(tmp_file.name, language=language)
        
        return result
    
    def get_segments_with_speakers(self, transcription_result: Dict, 
                                   diarization_result: List[Dict]) -> List[Dict]:
        """
        Merge transcription with speaker diarization results.
        
        Args:
            transcription_result: Output from transcribe()
            diarization_result: Output from SpeakerDiarizer
            
        Returns:
            List of segments with speaker labels
        """
        merged_segments = []
        
        for seg in transcription_result['segments']:
            seg_start = seg['start']
            seg_end = seg['end']
            
            # Find overlapping speaker segments
            speakers = []
            for diar_seg in diarization_result:
                if (diar_seg['start'] < seg_end and diar_seg['end'] > seg_start):
                    overlap = min(seg_end, diar_seg['end']) - max(seg_start, diar_seg['start'])
                    total_duration = seg_end - seg_start
                    if overlap / total_duration > 0.5:  # More than 50% overlap
                        speakers.append(diar_seg['speaker'])
            
            if speakers:
                # Most common speaker in this segment
                speaker = max(set(speakers), key=speakers.count)
            else:
                speaker = "UNKNOWN"
            
            merged_segments.append({
                **seg,
                'speaker': speaker
            })
        
        return merged_segments


if __name__ == "__main__":
    # Example usage
    print("Speech-to-Text Transcriber Module")
    print("=" * 50)
    print("This module uses Faster-Whisper for transcription.")
    print("Initialize with: transcriber = SpeechToTextTranscriber(model_size='base')")
    print("Transcribe with: result = transcriber.transcribe('audio.wav')")
