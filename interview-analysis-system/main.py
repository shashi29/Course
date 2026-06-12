"""
Interview Analysis System - Main Integration Module

This is the main entry point for the interview analysis system that combines
all audio and video processing capabilities to analyze candidate interviews.

Usage:
    python main.py --audio interview.wav --video interview.mp4 --output results.json
"""

import argparse
import json
import time
from typing import Dict, Optional
from datetime import datetime


class InterviewAnalyzer:
    """
    Main class that integrates all analysis modules for comprehensive
    interview analysis including audio and video processing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the interview analyzer with all required modules.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.results = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'config': self.config
            }
        }
        
        print("=" * 60)
        print("INTERVIEW ANALYSIS SYSTEM")
        print("=" * 60)
        print("\nInitializing analysis modules...\n")
    
    def initialize_audio_modules(self, use_gpu: bool = False):
        """Initialize all audio processing modules."""
        device = "cuda" if use_gpu else "cpu"
        
        print("Audio Processing Modules:")
        print("-" * 40)
        
        try:
            from audio_processing.transcriber import SpeechToTextTranscriber
            self.transcriber = SpeechToTextTranscriber(
                model_size=self.config.get('transcription_model', 'base'),
                device=device
            )
            print("✓ Speech-to-Text (Faster-Whisper)")
        except Exception as e:
            print(f"✗ Speech-to-Text: {e}")
            self.transcriber = None
        
        try:
            from audio_processing.diarizer import SpeakerDiarizer
            token = self.config.get('huggingface_token')
            self.diarizer = SpeakerDiarizer(use_auth_token=token)
            print("✓ Speaker Diarization (PyAnnote)")
        except Exception as e:
            print(f"✗ Speaker Diarization: {e}")
            self.diarizer = None
        
        try:
            from audio_processing.voice_embeddings import VoiceEmbeddingExtractor
            self.voice_extractor = VoiceEmbeddingExtractor(device=device)
            print("✓ Voice Embeddings (SpeechBrain)")
        except Exception as e:
            print(f"✗ Voice Embeddings: {e}")
            self.voice_extractor = None
        
        try:
            from audio_processing.audio_features import AudioFeatureExtractor
            self.feature_extractor = AudioFeatureExtractor()
            print("✓ Audio Features (Librosa)")
        except Exception as e:
            print(f"✗ Audio Features: {e}")
            self.feature_extractor = None
        
        try:
            from audio_processing.emotion_detector import EmotionDetector
            self.emotion_detector = EmotionDetector(device=device)
            print("✓ Emotion Detection (SpeechBrain)")
        except Exception as e:
            print(f"✗ Emotion Detection: {e}")
            self.emotion_detector = None
        
        print()
    
    def initialize_video_modules(self):
        """Initialize all video processing modules."""
        print("Video Processing Modules:")
        print("-" * 40)
        
        try:
            from video_processing.face_detector import FaceDetector
            self.face_detector = FaceDetector()
            print("✓ Face Detection (InsightFace)")
        except Exception as e:
            print(f"✗ Face Detection: {e}")
            self.face_detector = None
        
        try:
            from video_processing.face_landmarks import FaceLandmarkDetector
            self.landmark_detector = FaceLandmarkDetector()
            print("✓ Face Landmarks (MediaPipe)")
        except Exception as e:
            print(f"✗ Face Landmarks: {e}")
            self.landmark_detector = None
        
        try:
            from video_processing.gaze_estimator import GazeEstimator
            self.gaze_estimator = GazeEstimator()
            print("✓ Gaze Estimation (L2CS-Net)")
        except Exception as e:
            print(f"✗ Gaze Estimation: {e}")
            self.gaze_estimator = None
        
        try:
            from video_processing.head_pose import HeadPoseAnalyzer
            self.head_pose_analyzer = HeadPoseAnalyzer()
            print("✓ Head Pose Analysis (OpenCV)")
        except Exception as e:
            print(f"✗ Head Pose Analysis: {e}")
            self.head_pose_analyzer = None
        
        try:
            from video_processing.posture_analyzer import PostureAnalyzer
            self.posture_analyzer = PostureAnalyzer()
            print("✓ Posture Analysis (MediaPipe Pose)")
        except Exception as e:
            print(f"✗ Posture Analysis: {e}")
            self.posture_analyzer = None
        
        try:
            from video_processing.facial_behavior import FacialBehaviorAnalyzer
            self.facial_analyzer = FacialBehaviorAnalyzer()
            print("✓ Facial Behavior (Py-Feat)")
        except Exception as e:
            print(f"✗ Facial Behavior: {e}")
            self.facial_analyzer = None
        
        print()
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Perform complete audio analysis.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with all audio analysis results
        """
        print(f"\nAnalyzing audio: {audio_path}")
        print("=" * 50)
        
        audio_results = {}
        
        # Transcription
        if self.transcriber:
            print("  → Transcribing speech...")
            start = time.time()
            transcription = self.transcriber.transcribe(audio_path)
            audio_results['transcription'] = transcription
            print(f"    Completed in {time.time()-start:.2f}s")
            print(f"    Transcript length: {len(transcription['transcript'])} chars")
        
        # Speaker Diarization
        if self.diarizer and audio_path:
            print("  → Identifying speakers...")
            start = time.time()
            diarization = self.diarizer.diarize(audio_path)
            speaker_stats = self.diarizer.get_speaker_statistics(diarization)
            audio_results['speaker_diarization'] = {
                'segments': diarization,
                'statistics': speaker_stats
            }
            print(f"    Completed in {time.time()-start:.2f}s")
            print(f"    Speakers detected: {speaker_stats['num_speakers']}")
        
        # Audio Features
        if self.feature_extractor:
            print("  → Extracting audio features...")
            start = time.time()
            transcription_segments = audio_results.get('transcription', {}).get('segments', [])
            features = self.feature_extractor.extract_all_features(
                audio_path, 
                transcription_segments
            )
            audio_results['audio_features'] = features
            print(f"    Completed in {time.time()-start:.2f}s")
        
        # Emotion Detection
        if self.emotion_detector:
            print("  → Detecting emotions...")
            start = time.time()
            emotion = self.emotion_detector.detect_emotion(audio_path)
            
            # Calculate engagement if we have timeline
            if self.feature_extractor and audio_results.get('audio_features'):
                stress = self.emotion_detector.detect_stress_indicators(
                    audio_results['audio_features']
                )
                emotion['stress_analysis'] = stress
            
            audio_results['emotion'] = emotion
            print(f"    Completed in {time.time()-start:.2f}s")
            print(f"    Detected emotion: {emotion.get('predicted_emotion', 'N/A')}")
        
        return audio_results
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        Perform complete video analysis.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with all video analysis results
        """
        print(f"\nAnalyzing video: {video_path}")
        print("=" * 50)
        
        video_results = {}
        
        # For now, return placeholder - full implementation would process video frames
        video_results['status'] = 'Video analysis requires frame-by-frame processing'
        video_results['modules_available'] = {
            'face_detection': self.face_detector is not None,
            'face_landmarks': self.landmark_detector is not None,
            'gaze_estimation': self.gaze_estimator is not None,
            'head_pose': self.head_pose_analyzer is not None,
            'posture': self.posture_analyzer is not None,
            'facial_behavior': self.facial_analyzer is not None
        }
        
        return video_results
    
    def generate_report(self, audio_results: Dict, video_results: Dict) -> Dict:
        """
        Generate comprehensive analysis report.
        
        Args:
            audio_results: Results from audio analysis
            video_results: Results from video analysis
            
        Returns:
            Complete analysis report
        """
        report = {
            'summary': {},
            'audio_analysis': audio_results,
            'video_analysis': video_results,
            'recommendations': []
        }
        
        # Generate summary metrics
        if audio_results:
            summary = []
            
            # Transcription quality
            if 'transcription' in audio_results:
                transcript = audio_results['transcription']
                summary.append(f"Transcription: {len(transcript['transcript'])} characters")
            
            # Speaker information
            if 'speaker_diarization' in audio_results:
                stats = audio_results['speaker_diarization']['statistics']
                summary.append(f"Speakers: {stats['num_speakers']} detected")
            
            # Speaking rate
            if 'audio_features' in audio_results:
                features = audio_results['audio_features']
                if 'wpm' in features:
                    summary.append(f"Speaking rate: {features['wpm']['wpm']:.1f} WPM")
            
            # Emotion
            if 'emotion' in audio_results:
                emotion = audio_results['emotion']
                summary.append(f"Primary emotion: {emotion.get('predicted_emotion', 'N/A')}")
            
            report['summary']['key_metrics'] = summary
        
        # Generate recommendations
        recommendations = []
        
        if audio_results.get('audio_features', {}).get('wpm', {}).get('wpm', 150) > 180:
            recommendations.append("Consider slowing down speaking pace (currently >180 WPM)")
        
        if audio_results.get('audio_features', {}).get('silence_pauses', {}).get('silence_ratio', 0) > 0.3:
            recommendations.append("Reduce pauses and filler silence")
        
        if audio_results.get('emotion', {}).get('stress_analysis', {}).get('stress_level') == 'high':
            recommendations.append("High stress indicators detected - practice relaxation techniques")
        
        report['recommendations'] = recommendations
        
        return report
    
    def save_results(self, report: Dict, output_path: str):
        """Save analysis results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point for the interview analysis system."""
    parser = argparse.ArgumentParser(
        description='Interview Analysis System - Comprehensive audio/video analysis'
    )
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--output', type=str, default='analysis_results.json',
                       help='Output path for results')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--hf-token', type=str, help='HuggingFace token for PyAnnote')
    
    args = parser.parse_args()
    
    # Configuration
    config = {}
    if args.hf_token:
        config['huggingface_token'] = args.hf_token
    
    # Initialize analyzer
    analyzer = InterviewAnalyzer(config=config)
    
    # Initialize modules
    analyzer.initialize_audio_modules(use_gpu=args.gpu)
    analyzer.initialize_video_modules()
    
    # Perform analysis
    audio_results = {}
    video_results = {}
    
    if args.audio:
        audio_results = analyzer.analyze_audio(args.audio)
    
    if args.video:
        video_results = analyzer.analyze_video(args.video)
    
    # Generate and save report
    report = analyzer.generate_report(audio_results, video_results)
    analyzer.save_results(report, args.output)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Print summary
    if report['summary'].get('key_metrics'):
        print("\nKey Metrics:")
        for metric in report['summary']['key_metrics']:
            print(f"  • {metric}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
