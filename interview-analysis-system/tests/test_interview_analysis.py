"""
Unit Tests for Interview Analysis System

This module contains comprehensive unit tests for all components of the
Interview Analysis System including audio processing and video processing modules.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import warnings
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

# Suppress warnings during testing
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestSpeechToTextTranscriber(unittest.TestCase):
    """Tests for the Speech-to-Text Transcriber module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 2  # seconds
        self.audio_array = np.random.randn(self.sample_rate * self.duration).astype(np.float32)
        
    @patch('audio_processing.transcriber.WhisperModel')
    def test_transcriber_initialization(self, mock_whisper):
        """Test transcriber initialization with different model sizes."""
        from audio_processing.transcriber import SpeechToTextTranscriber
        
        # Mock the model instance
        mock_instance = Mock()
        mock_whisper.return_value = mock_instance
        
        transcriber = SpeechToTextTranscriber(model_size='base')
        
        self.assertEqual(transcriber.model_size, 'base')
        self.assertEqual(transcriber.device, 'auto')
        mock_whisper.assert_called_once()
    
    @patch('audio_processing.transcriber.WhisperModel')
    def test_transcribe_mock(self, mock_whisper):
        """Test transcription with mocked model."""
        from audio_processing.transcriber import SpeechToTextTranscriber
        
        # Setup mock
        mock_instance = Mock()
        mock_whisper.return_value = mock_instance
        
        # Mock segment
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.text = "Hello world"
        mock_segment.avg_logprob = -0.5
        mock_segment.words = []
        
        mock_info = Mock()
        mock_info.language = 'en'
        mock_info.language_probability = 0.99
        
        mock_instance.transcribe.return_value = ([mock_segment], mock_info)
        
        transcriber = SpeechToTextTranscriber(model_size='base')
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, self.audio_array, self.sample_rate)
            result = transcriber.transcribe(tmp.name)
        
        self.assertIn('transcript', result)
        self.assertIn('segments', result)
        self.assertIn('language', result)
        self.assertEqual(result['language'], 'en')
        
        # Cleanup
        os.unlink(tmp.name)
    
    def test_get_segments_with_speakers(self):
        """Test merging transcription with speaker diarization."""
        from audio_processing.transcriber import SpeechToTextTranscriber
        
        transcriber = SpeechToTextTranscriber.__new__(SpeechToTextTranscriber)
        
        transcription = {
            'segments': [
                {'start': 0.0, 'end': 2.0, 'text': 'Hello'},
                {'start': 2.5, 'end': 4.0, 'text': 'World'}
            ]
        }
        
        diarization = [
            {'start': 0.0, 'end': 2.5, 'speaker': 'SPEAKER_0'},
            {'start': 2.5, 'end': 5.0, 'speaker': 'SPEAKER_1'}
        ]
        
        merged = transcriber.get_segments_with_speakers(transcription, diarization)
        
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]['speaker'], 'SPEAKER_0')
        self.assertEqual(merged[1]['speaker'], 'SPEAKER_1')


class TestSpeakerDiarizer(unittest.TestCase):
    """Tests for the Speaker Diarization module."""
    
    @patch('audio_processing.diarizer.Pipeline')
    def test_diarizer_initialization(self, mock_pipeline):
        """Test diarizer initialization."""
        from audio_processing.diarizer import SpeakerDiarizer
        
        mock_pipe_instance = Mock()
        mock_pipeline.from_pretrained.return_value = mock_pipe_instance
        
        diarizer = SpeakerDiarizer(use_auth_token='test_token')
        
        self.assertEqual(diarizer.model_name, 'pyannote/speaker-diarization-3.1')
        mock_pipeline.from_pretrained.assert_called_once()
    
    @patch('audio_processing.diarizer.Pipeline')
    def test_get_speaker_statistics(self, mock_pipeline):
        """Test speaker statistics calculation."""
        from audio_processing.diarizer import SpeakerDiarizer
        
        diarizer = SpeakerDiarizer.__new__(SpeakerDiarizer)
        
        diarization_result = [
            {'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_0'},
            {'start': 2.0, 'end': 4.0, 'speaker': 'SPEAKER_1'},
            {'start': 4.0, 'end': 6.0, 'speaker': 'SPEAKER_0'}
        ]
        
        stats = diarizer.get_speaker_statistics(diarization_result)
        
        self.assertIn('num_speakers', stats)
        self.assertEqual(stats['num_speakers'], 2)
        self.assertIn('total_audio_duration', stats)
        self.assertAlmostEqual(stats['total_audio_duration'], 6.0, places=5)
    
    def test_get_speaker_timeline(self):
        """Test speaker timeline generation."""
        from audio_processing.diarizer import SpeakerDiarizer
        
        diarizer = SpeakerDiarizer.__new__(SpeakerDiarizer)
        
        diarization_result = [
            {'start': 0.0, 'end': 2.0, 'speaker': 'SPEAKER_0'},
            {'start': 2.0, 'end': 4.0, 'speaker': 'SPEAKER_1'}
        ]
        
        timeline = diarizer.get_speaker_timeline(diarization_result)
        
        self.assertIn('SPEAKER_0', timeline)
        self.assertIn('SPEAKER_1', timeline)
        self.assertEqual(len(timeline['SPEAKER_0']), 1)
        self.assertEqual(timeline['SPEAKER_0'][0]['duration'], 2.0)


class TestVoiceEmbeddingExtractor(unittest.TestCase):
    """Tests for the Voice Embeddings module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 2
        self.audio_array = np.random.randn(self.sample_rate * self.duration).astype(np.float32)
    
    @patch('audio_processing.voice_embeddings.EncoderClassifier')
    def test_extractor_initialization(self, mock_classifier):
        """Test extractor initialization."""
        from audio_processing.voice_embeddings import VoiceEmbeddingExtractor
        
        mock_instance = Mock()
        mock_classifier.from_hparams.return_value = mock_instance
        
        extractor = VoiceEmbeddingExtractor()
        
        self.assertEqual(extractor.model_name, 'speechbrain/spkrec-ecapa-voxceleb')
        mock_classifier.from_hparams.assert_called_once()
    
    @patch('audio_processing.voice_embeddings.EncoderClassifier')
    def test_verify_speakers_same(self, mock_classifier):
        """Test speaker verification with same speaker."""
        from audio_processing.voice_embeddings import VoiceEmbeddingExtractor
        
        mock_instance = Mock()
        mock_classifier.from_hparams.return_value = mock_instance
        
        # Mock embedding extraction
        mock_embedding = np.random.randn(192).astype(np.float32)
        mock_instance.encode_batch.return_value = mock_embedding[np.newaxis, np.newaxis, :]
        
        extractor = VoiceEmbeddingExtractor()
        
        # Create temporary audio files
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp1:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp2:
                import soundfile as sf
                sf.write(tmp1.name, self.audio_array, self.sample_rate)
                sf.write(tmp2.name, self.audio_array, self.sample_rate)
                
                result = extractor.verify_speakers(tmp1.name, tmp2.name)
        
        self.assertIn('is_same_speaker', result)
        self.assertIn('similarity_score', result)
        self.assertIn('threshold', result)
        
        # Cleanup
        os.unlink(tmp1.name)
        os.unlink(tmp2.name)


class TestAudioFeatureExtractor(unittest.TestCase):
    """Tests for the Audio Feature Extraction module."""
    
    def setUp(self):
        """Set up test fixtures."""
        from audio_processing.audio_features import AudioFeatureExtractor
        self.extractor = AudioFeatureExtractor(sample_rate=16000)
        self.sample_rate = 16000
        self.duration = 2
        
    def test_extract_pitch(self):
        """Test pitch extraction."""
        # Generate a simple sine wave (440 Hz)
        t = np.linspace(0, self.duration, self.sample_rate * self.duration)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        pitch_result = self.extractor.extract_pitch(audio, self.sample_rate)
        
        self.assertIn('mean_pitch', pitch_result)
        self.assertIn('std_pitch', pitch_result)
        self.assertIn('min_pitch', pitch_result)
        self.assertIn('max_pitch', pitch_result)
        self.assertIn('pitch_range', pitch_result)
        
        # Mean pitch should be close to 440 Hz for a pure tone
        if pitch_result['voiced_ratio'] > 0:
            self.assertGreater(pitch_result['mean_pitch'], 400)
            self.assertLess(pitch_result['mean_pitch'], 480)
    
    def test_extract_energy(self):
        """Test energy extraction."""
        audio = np.random.randn(self.sample_rate * self.duration).astype(np.float32)
        
        energy_result = self.extractor.extract_energy(audio)
        
        self.assertIn('mean_energy', energy_result)
        self.assertIn('std_energy', energy_result)
        self.assertIn('min_energy', energy_result)
        self.assertIn('max_energy', energy_result)
        self.assertGreater(energy_result['mean_energy'], 0)
    
    def test_detect_silence_and_pauses(self):
        """Test silence detection."""
        # Create audio with silence
        audio = np.zeros(self.sample_rate)  # 1 second of silence
        speech = np.random.randn(self.sample_rate).astype(np.float32)  # 1 second of noise
        audio_with_speech = np.concatenate([audio, speech])
        
        silence_result = self.extractor.detect_silence_and_pauses(
            audio_with_speech, self.sample_rate, threshold_db=-40
        )
        
        self.assertIn('total_silence_duration', silence_result)
        self.assertIn('silence_ratio', silence_result)
        self.assertIn('num_silent_segments', silence_result)
        self.assertGreater(silence_result['silence_ratio'], 0.3)
    
    def test_estimate_wpm(self):
        """Test WPM estimation."""
        segments = [
            {'text': 'Hello world this is a test'},
            {'text': 'Another segment of text'}
        ]
        
        wpm_result = self.extractor.estimate_wpm(segments, audio_duration=30.0)
        
        self.assertIn('total_words', wpm_result)
        self.assertIn('wpm', wpm_result)
        self.assertEqual(wpm_result['total_words'], 11)
        self.assertGreater(wpm_result['wpm'], 0)
    
    def test_extract_all_features(self):
        """Test extracting all features together."""
        audio = np.random.randn(self.sample_rate * self.duration).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, self.sample_rate)
            
            features = self.extractor.extract_all_features(tmp.name)
        
        self.assertIn('pitch', features)
        self.assertIn('energy', features)
        self.assertIn('spectral', features)
        self.assertIn('silence_pauses', features)
        self.assertIn('audio_info', features)
        
        # Cleanup
        os.unlink(tmp.name)


class TestEmotionDetector(unittest.TestCase):
    """Tests for the Emotion Detection module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.duration = 2
        self.audio_array = np.random.randn(self.sample_rate * self.duration).astype(np.float32)
    
    @patch('audio_processing.emotion_detector.EncoderClassifier')
    def test_detector_initialization(self, mock_classifier):
        """Test emotion detector initialization."""
        from audio_processing.emotion_detector import EmotionDetector
        
        mock_instance = Mock()
        mock_classifier.from_hparams.return_value = mock_instance
        
        detector = EmotionDetector()
        
        self.assertEqual(detector.model_name, 'speechbrain/emotion-recognition-wav2vec2-MSP')
    
    def test_calculate_engagement_score(self):
        """Test engagement score calculation."""
        from audio_processing.emotion_detector import EmotionDetector
        
        detector = EmotionDetector.__new__(EmotionDetector)
        
        timeline = [
            {'predicted_emotion': 'happy', 'confidence': 0.8},
            {'predicted_emotion': 'excited', 'confidence': 0.7},
            {'predicted_emotion': 'neutral', 'confidence': 0.6},
            {'predicted_emotion': 'sad', 'confidence': 0.5}
        ]
        
        engagement = detector.calculate_engagement_score(timeline)
        
        self.assertIn('engagement_score', engagement)
        self.assertIn('positive_ratio', engagement)
        self.assertIn('negative_ratio', engagement)
        self.assertGreaterEqual(engagement['engagement_score'], 0)
        self.assertLessEqual(engagement['engagement_score'], 1)
    
    def test_detect_stress_indicators(self):
        """Test stress indicator detection."""
        from audio_processing.emotion_detector import EmotionDetector
        
        detector = EmotionDetector.__new__(EmotionDetector)
        
        audio_features = {
            'pitch': {'std_pitch': 60},  # High variability
            'wpm': {'wpm': 200},  # Fast speaking
            'silence_pauses': {'silence_ratio': 0.4},  # High silence
            'energy': {'mean_energy': 0.03}  # Low energy
        }
        
        stress_result = detector.detect_stress_indicators(audio_features)
        
        self.assertIn('stress_score', stress_result)
        self.assertIn('stress_level', stress_result)
        self.assertIn('indicators', stress_result)
        self.assertGreater(len(stress_result['indicators']), 0)


class TestFaceDetector(unittest.TestCase):
    """Tests for the Face Detection module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple rectangle to simulate a face region
        cv2.rectangle(self.frame, (100, 100), (200, 200), (255, 255, 255), -1)
    
    @patch('video_processing.face_detector.FaceAnalysis')
    def test_detector_initialization(self, mock_face_analysis):
        """Test face detector initialization."""
        from video_processing.face_detector import FaceDetector
        
        mock_app = Mock()
        mock_face_analysis.return_value = mock_app
        
        detector = FaceDetector()
        
        self.assertIsNotNone(detector.app)
        mock_app.prepare.assert_called_once()
    
    def test_count_faces_empty(self):
        """Test counting faces in empty frame (without actual model)."""
        from video_processing.face_detector import FaceDetector
        
        detector = FaceDetector.__new__(FaceDetector)
        detector.app = None
        
        # Without model, should handle gracefully
        # This tests the structure rather than actual detection
        self.assertTrue(hasattr(detector, 'count_faces'))


class TestFaceLandmarkDetector(unittest.TestCase):
    """Tests for the Face Landmark Detection module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    @patch('video_processing.face_landmarks.mp.solutions.face_mesh.FaceMesh')
    def test_landmark_detector_initialization(self, mock_facemesh):
        """Test landmark detector initialization."""
        from video_processing.face_landmarks import FaceLandmarkDetector
        
        mock_instance = Mock()
        mock_facemesh.return_value = mock_instance
        
        detector = FaceLandmarkDetector()
        
        self.assertIsNotNone(detector.face_mesh)
    
    def test_calculate_eye_aspect_ratio(self):
        """Test eye aspect ratio calculation."""
        from video_processing.face_landmarks import FaceLandmarkDetector
        
        detector = FaceLandmarkDetector.__new__(FaceLandmarkDetector)
        
        # Simulate open eye landmarks (6 points)
        open_eye = [
            [0.3, 0.3, 0],  # Left corner
            [0.35, 0.28, 0],  # Top
            [0.4, 0.28, 0],  # Top-right
            [0.45, 0.3, 0],  # Right corner
            [0.4, 0.32, 0],  # Bottom-right
            [0.35, 0.32, 0]   # Bottom
        ]
        
        ear_open = detector.calculate_eye_aspect_ratio(open_eye)
        self.assertGreater(ear_open, 0)
        
        # Simulate closed eye (vertical distance near zero)
        closed_eye = [
            [0.3, 0.3, 0],
            [0.35, 0.3, 0],  # Top same as bottom
            [0.4, 0.3, 0],
            [0.45, 0.3, 0],
            [0.4, 0.3, 0],
            [0.35, 0.3, 0]
        ]
        
        ear_closed = detector.calculate_eye_aspect_ratio(closed_eye)
        self.assertAlmostEqual(ear_closed, 0, places=5)
    
    def test_detect_blinks(self):
        """Test blink detection."""
        from video_processing.face_landmarks import FaceLandmarkDetector
        
        detector = FaceLandmarkDetector.__new__(FaceLandmarkDetector)
        
        # Mock landmark results with low EAR (blink)
        landmark_results = [{
            'left_eye': [[0.3, 0.3, 0]] * 6,
            'right_eye': [[0.4, 0.3, 0]] * 6
        }]
        
        blink_result = detector.detect_blinks(landmark_results, threshold=0.25)
        
        self.assertIn('left_eye_blink', blink_result)
        self.assertIn('right_eye_blink', blink_result)
        self.assertIn('avg_ear', blink_result)


class TestGazeEstimator(unittest.TestCase):
    """Tests for the Gaze Estimation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.face_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
    
    def test_gaze_estimator_initialization(self):
        """Test gaze estimator initialization."""
        from video_processing.gaze_estimator import GazeEstimator
        
        estimator = GazeEstimator()
        
        self.assertEqual(estimator.device, 'cpu')
        self.assertIsNone(estimator.model)  # Uses fallback
    
    def test_estimate_gaze_from_landmarks(self):
        """Test gaze estimation from landmarks."""
        from video_processing.gaze_estimator import GazeEstimator
        
        estimator = GazeEstimator()
        
        # Create mock landmarks (468 points)
        landmarks = [[0.5, 0.5, 0]] * 468
        
        result = estimator._estimate_gaze_from_landmarks(landmarks, self.face_bbox)
        
        self.assertIn('yaw', result)
        self.assertIn('pitch', result)
        self.assertIn('eye_contact', result)
        self.assertIn('confidence', result)
    
    def test_analyze_eye_contact(self):
        """Test eye contact analysis."""
        from video_processing.gaze_estimator import GazeEstimator
        
        estimator = GazeEstimator()
        
        gaze_results = [
            {'eye_contact': True, 'yaw': 5, 'pitch': 3},
            {'eye_contact': True, 'yaw': 8, 'pitch': 5},
            {'eye_contact': False, 'yaw': 30, 'pitch': 20},
            {'eye_contact': True, 'yaw': 10, 'pitch': 8}
        ]
        
        analysis = estimator.analyze_eye_contact(gaze_results)
        
        self.assertIn('eye_contact_ratio', analysis)
        self.assertIn('avg_eye_contact_duration', analysis)
        self.assertIn('num_eye_contact_periods', analysis)
        self.assertAlmostEqual(analysis['eye_contact_ratio'], 0.75, places=2)
    
    def test_get_attention_score(self):
        """Test attention score calculation."""
        from video_processing.gaze_estimator import GazeEstimator
        
        estimator = GazeEstimator()
        
        gaze_results = [
            {'yaw': 5, 'pitch': 3},  # Good attention
            {'yaw': 10, 'pitch': 8},  # Good attention
            {'yaw': 45, 'pitch': 30}  # Poor attention
        ]
        
        score = estimator.get_attention_score(gaze_results)
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertGreater(score, 0.3)  # Should be reasonably high


class TestHeadPoseAnalyzer(unittest.TestCase):
    """Tests for the Head Pose Analysis module."""
    
    def setUp(self):
        """Set up test fixtures."""
        from video_processing.head_pose import HeadPoseAnalyzer
        self.analyzer = HeadPoseAnalyzer()
        self.frame_shape = (480, 640, 3)
    
    def test_head_pose_analyzer_initialization(self):
        """Test head pose analyzer initialization."""
        self.assertIsNotNone(self.analyzer.camera_matrix)
        self.assertIsNotNone(self.analyzer.dist_coeffs)
    
    def test_estimate_head_pose(self):
        """Test head pose estimation."""
        # Create mock landmarks (468 points)
        landmarks = [[0.5, 0.5, 0]] * 468
        
        pose = self.analyzer.estimate_head_pose(landmarks, self.frame_shape)
        
        self.assertIn('yaw', pose)
        self.assertIn('pitch', pose)
        self.assertIn('roll', pose)
        self.assertIn('rotation_vector', pose)
        self.assertIn('translation_vector', pose)
    
    def test_detect_reading_behavior(self):
        """Test reading behavior detection."""
        pose_results = [
            {'yaw': 5, 'pitch': 25},  # Looking down
            {'yaw': 3, 'pitch': 10},  # Normal
            {'yaw': 35, 'pitch': 5},  # Looking away
            {'yaw': 2, 'pitch': 8}    # Normal
        ]
        
        behavior = self.analyzer.detect_reading_behavior(pose_results)
        
        self.assertIn('reading_ratio', behavior)
        self.assertIn('looking_down_ratio', behavior)
        self.assertIn('looking_away_ratio', behavior)
        self.assertIn('attentive_ratio', behavior)
    
    def test_calculate_head_movement(self):
        """Test head movement calculation."""
        pose_results = [
            {'yaw': 0, 'pitch': 0, 'roll': 0},
            {'yaw': 10, 'pitch': 5, 'roll': 2},
            {'yaw': 5, 'pitch': 3, 'roll': 1},
            {'yaw': 15, 'pitch': 8, 'roll': 3}
        ]
        
        movement = self.analyzer.calculate_head_movement(pose_results)
        
        self.assertIn('avg_yaw_change', movement)
        self.assertIn('avg_pitch_change', movement)
        self.assertIn('avg_roll_change', movement)
        self.assertGreater(movement['avg_yaw_change'], 0)


class TestPostureAnalyzer(unittest.TestCase):
    """Tests for the Posture Analysis module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    @patch('video_processing.posture_analyzer.mp.solutions.pose.Pose')
    def test_posture_analyzer_initialization(self, mock_pose):
        """Test posture analyzer initialization."""
        from video_processing.posture_analyzer import PostureAnalyzer
        
        mock_instance = Mock()
        mock_pose.return_value = mock_instance
        
        analyzer = PostureAnalyzer()
        
        self.assertIsNotNone(analyzer.pose)
    
    def test_calculate_posture_angles(self):
        """Test posture angle calculation."""
        from video_processing.posture_analyzer import PostureAnalyzer
        
        analyzer = PostureAnalyzer.__new__(PostureAnalyzer)
        
        # Create mock landmarks (33 points for MediaPipe Pose)
        landmarks = []
        for i in range(33):
            landmarks.append({
                'x': 320 + np.random.randn() * 10,
                'y': 240 + np.random.randn() * 10,
                'z': np.random.randn(),
                'visibility': 0.9
            })
        
        angles = analyzer.calculate_posture_angles(landmarks)
        
        self.assertIn('torso_angle', angles)
        self.assertIn('shoulder_slope', angles)
        self.assertIn('head_forward_offset', angles)
        self.assertIn('is_upright', angles)
        self.assertIn('is_slouching', angles)
    
    def test_analyze_sitting_posture(self):
        """Test sitting posture analysis over time."""
        from video_processing.posture_analyzer import PostureAnalyzer
        
        analyzer = PostureAnalyzer.__new__(PostureAnalyzer)
        
        pose_results = [
            {'has_pose': True, 'landmarks': [{'x': 320, 'y': 240, 'z': 0, 'visibility': 0.9}] * 33},
            {'has_pose': True, 'landmarks': [{'x': 320, 'y': 240, 'z': 0, 'visibility': 0.9}] * 33},
            {'has_pose': False, 'landmarks': []}
        ]
        
        analysis = analyzer.analyze_sitting_posture(pose_results)
        
        self.assertIn('avg_torso_angle', analysis)
        self.assertIn('slouching_ratio', analysis)
        self.assertIn('good_posture_ratio', analysis)
        self.assertIn('frames_analyzed', analysis)


class TestFacialBehaviorAnalyzer(unittest.TestCase):
    """Tests for the Facial Behavior Analysis module."""
    
    def setUp(self):
        """Set up test fixtures."""
        from video_processing.facial_behavior import FacialBehaviorAnalyzer
        self.analyzer = FacialBehaviorAnalyzer()
    
    def test_facial_behavior_analyzer_initialization(self):
        """Test facial behavior analyzer initialization."""
        self.assertIsNotNone(self.analyzer)
    
    def test_analyze_facial_expression_insufficient_landmarks(self):
        """Test expression analysis with insufficient landmarks."""
        landmarks = [[0.5, 0.5, 0]] * 100  # Less than 468
        
        result = self.analyzer.analyze_facial_expression(landmarks)
        
        self.assertEqual(result['dominant_expression'], 'neutral')
        self.assertEqual(result['engagement_level'], 0.5)
    
    def test_analyze_facial_expression_full_landmarks(self):
        """Test expression analysis with full landmarks."""
        # Create 468 landmarks
        landmarks = []
        for i in range(468):
            x = 0.5 + np.random.randn() * 0.1
            y = 0.5 + np.random.randn() * 0.1
            z = np.random.randn() * 0.01
            landmarks.append([x, y, z])
        
        result = self.analyzer.analyze_facial_expression(landmarks)
        
        self.assertIn('dominant_expression', result)
        self.assertIn('expression_scores', result)
        self.assertIn('mouth', result)
        self.assertIn('eyes', result)
        self.assertIn('brows', result)
        self.assertIn('engagement_level', result)
        
        # Check expression scores sum to ~1
        total = sum(result['expression_scores'].values())
        self.assertAlmostEqual(total, 1.0, places=1)
    
    def test_analyze_attention_timeline(self):
        """Test attention timeline analysis."""
        expression_results = [
            {'engagement_level': 0.8, 'dominant_expression': 'focused'},
            {'engagement_level': 0.7, 'dominant_expression': 'focused'},
            {'engagement_level': 0.3, 'dominant_expression': 'neutral'},
            {'engagement_level': 0.9, 'dominant_expression': 'focused'},
            {'engagement_level': 0.85, 'dominant_expression': 'focused'}
        ]
        
        attention = self.analyzer.analyze_attention_timeline(expression_results)
        
        self.assertIn('avg_engagement', attention)
        self.assertIn('max_engagement', attention)
        self.assertIn('attention_span_frames', attention)
        self.assertIn('focused_ratio', attention)
        
        self.assertGreater(attention['avg_engagement'], 0.5)
        self.assertEqual(attention['focused_ratio'], 0.8)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_audio_processing_pipeline(self):
        """Test complete audio processing pipeline."""
        from audio_processing.audio_features import AudioFeatureExtractor
        from audio_processing.emotion_detector import EmotionDetector
        
        # Create sample audio
        sample_rate = 16000
        duration = 2
        audio = np.random.randn(sample_rate * duration).astype(np.float32)
        
        # Extract features
        feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio, sample_rate)
            
            features = feature_extractor.extract_all_features(tmp.name)
            
            # Test emotion detection fallback
            detector = EmotionDetector.__new__(EmotionDetector)
            detector.classifier = None
            emotion_result = detector._rule_based_emotion_detection(tmp.name)
        
        self.assertIn('pitch', features)
        self.assertIn('energy', features)
        self.assertIn('predicted_emotion', emotion_result)
        
        # Cleanup
        os.unlink(tmp.name)
    
    def test_video_processing_pipeline(self):
        """Test complete video processing pipeline."""
        from video_processing.gaze_estimator import GazeEstimator
        from video_processing.head_pose import HeadPoseAnalyzer
        from video_processing.facial_behavior import FacialBehaviorAnalyzer
        
        # Create mock data
        frame_shape = (480, 640, 3)
        landmarks = [[0.5, 0.5, 0]] * 468
        face_bbox = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}
        
        # Process through multiple analyzers
        gaze_estimator = GazeEstimator()
        gaze_result = gaze_estimator._estimate_gaze_from_landmarks(landmarks, face_bbox)
        
        head_pose_analyzer = HeadPoseAnalyzer()
        pose_result = head_pose_analyzer.estimate_head_pose(landmarks, frame_shape)
        
        facial_analyzer = FacialBehaviorAnalyzer()
        expression_result = facial_analyzer.analyze_facial_expression(landmarks)
        
        # Verify all outputs
        self.assertIn('eye_contact', gaze_result)
        self.assertIn('yaw', pose_result)
        self.assertIn('engagement_level', expression_result)
    
    def test_combined_metrics_calculation(self):
        """Test combining metrics from multiple sources."""
        from video_processing.gaze_estimator import GazeEstimator
        from audio_processing.emotion_detector import EmotionDetector
        
        # Simulate gaze results
        gaze_results = [
            {'eye_contact': True, 'yaw': 5, 'pitch': 3},
            {'eye_contact': True, 'yaw': 8, 'pitch': 5},
            {'eye_contact': False, 'yaw': 30, 'pitch': 20}
        ]
        
        # Simulate emotion timeline
        emotion_timeline = [
            {'predicted_emotion': 'confident', 'confidence': 0.8},
            {'predicted_emotion': 'happy', 'confidence': 0.7},
            {'predicted_emotion': 'neutral', 'confidence': 0.6}
        ]
        
        # Calculate combined metrics
        gaze_estimator = GazeEstimator()
        attention_score = gaze_estimator.get_attention_score(gaze_results)
        
        emotion_detector = EmotionDetector.__new__(EmotionDetector)
        engagement_score = emotion_detector.calculate_engagement_score(emotion_timeline)
        
        # Combined score
        combined_score = (attention_score + engagement_score['engagement_score']) / 2
        
        self.assertGreaterEqual(combined_score, 0)
        self.assertLessEqual(combined_score, 1)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSpeechToTextTranscriber,
        TestSpeakerDiarizer,
        TestVoiceEmbeddingExtractor,
        TestAudioFeatureExtractor,
        TestEmotionDetector,
        TestFaceDetector,
        TestFaceLandmarkDetector,
        TestGazeEstimator,
        TestHeadPoseAnalyzer,
        TestPostureAnalyzer,
        TestFacialBehaviorAnalyzer,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("=" * 70)
    print("INTERVIEW ANALYSIS SYSTEM - UNIT TESTS")
    print("=" * 70)
    print("\nRunning comprehensive unit tests for all modules...\n")
    
    result = run_tests()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 70)
    
    sys.exit(0 if result.wasSuccessful() else 1)
