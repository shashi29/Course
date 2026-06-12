# Interview Analysis System

A comprehensive open-source interview analysis system that uses state-of-the-art AI models to analyze candidate interviews through audio and video processing.

## Features

### Audio Processing
| Layer | Tool | Purpose |
|-------|------|---------|
| Speech-to-Text | [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) | Fastest production Whisper implementation for transcription |
| Speaker Diarization | [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) | Best open-source speaker identification |
| Voice Embeddings | [SpeechBrain](https://github.com/speechbrain/speechbrain) | Speaker recognition and verification |
| Audio Features | [OpenSMILE](https://github.com/audeering/opensmile-python) + [Librosa](https://librosa.org/) | Pitch, tone, energy, silence, WPM extraction |
| Emotion Detection | [SpeechBrain Emotion Models](https://github.com/speechbrain/speechbrain) | Engagement, stress, confidence detection |

### Video Processing
| Layer | Tool | Purpose |
|-------|------|---------|
| Face Detection | [InsightFace](https://github.com/deepinsight/insightface) | Fast and accurate face analysis |
| Face Landmarks | [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh) | 468 facial landmarks for eye tracking |
| Eye Contact | [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) | State-of-the-art gaze estimation |
| Head Pose | MediaPipe + OpenCV | Reading-screen detection, head orientation |
| Posture Analysis | [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose) | Body landmark tracking for sitting posture |
| Facial Behavior | [Py-Feat](https://github.com/DillonDuPont/py-feat) | Research-grade facial analysis for attention/engagement |

## Installation

```bash
cd interview-analysis-system
pip install -r requirements.txt
```

### Important Notes

1. **HuggingFace Token**: For PyAnnote speaker diarization, you need a HuggingFace token:
   - Visit: https://huggingface.co/settings/tokens
   - Create a new token with "read" permissions
   - Use it with: `python main.py --hf-token YOUR_TOKEN`

2. **GPU Acceleration**: For faster processing, install CUDA-enabled versions:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

### Basic Usage

```bash
# Analyze audio only
python main.py --audio interview.wav --output results.json

# Analyze video only
python main.py --video interview.mp4 --output results.json

# Analyze both audio and video
python main.py --audio interview.wav --video interview.mp4 --output results.json

# With GPU acceleration
python main.py --audio interview.wav --gpu --output results.json

# With HuggingFace token for speaker diarization
python main.py --audio interview.wav --hf-token YOUR_TOKEN --output results.json
```

### Programmatic Usage

```python
from main import InterviewAnalyzer

# Initialize analyzer
analyzer = InterviewAnalyzer()

# Initialize modules
analyzer.initialize_audio_modules(use_gpu=False)
analyzer.initialize_video_modules()

# Perform analysis
audio_results = analyzer.analyze_audio('interview.wav')
video_results = analyzer.analyze_video('interview.mp4')

# Generate report
report = analyzer.generate_report(audio_results, video_results)
analyzer.save_results(report, 'results.json')
```

## Output Format

The analysis results are saved as JSON with the following structure:

```json
{
  "summary": {
    "key_metrics": [
      "Transcription: 5000 characters",
      "Speakers: 2 detected",
      "Speaking rate: 150.5 WPM",
      "Primary emotion: neutral"
    ]
  },
  "audio_analysis": {
    "transcription": {...},
    "speaker_diarization": {...},
    "audio_features": {...},
    "emotion": {...}
  },
  "video_analysis": {
    "modules_available": {...}
  },
  "recommendations": [
    "Consider slowing down speaking pace",
    "Reduce pauses and filler silence"
  ]
}
```

## Module Structure

```
interview-analysis-system/
├── audio_processing/
│   ├── __init__.py
│   ├── transcriber.py        # Faster-Whisper
│   ├── diarizer.py           # PyAnnote Audio
│   ├── voice_embeddings.py   # SpeechBrain
│   ├── audio_features.py     # Librosa/OpenSMILE
│   └── emotion_detector.py   # SpeechBrain Emotion
├── video_processing/
│   ├── __init__.py
│   ├── face_detector.py      # InsightFace
│   ├── face_landmarks.py     # MediaPipe Face Mesh
│   ├── gaze_estimator.py     # L2CS-Net
│   ├── head_pose.py          # OpenCV
│   ├── posture_analyzer.py   # MediaPipe Pose
│   └── facial_behavior.py    # Py-Feat
├── main.py                   # Main integration module
├── requirements.txt
└── README.md
```

## Metrics Analyzed

### Audio Metrics
- **Transcription accuracy** - Full speech-to-text conversion
- **Speaker identification** - Who spoke when
- **Speaking rate (WPM)** - Words per minute
- **Pitch variation** - Voice tone analysis
- **Energy levels** - Speech intensity
- **Silence ratio** - Pause detection
- **Emotional state** - Stress, engagement, confidence

### Video Metrics
- **Eye contact ratio** - Gaze estimation
- **Head movement** - Yaw, pitch, roll angles
- **Reading detection** - Looking down at notes
- **Posture quality** - Slouching detection
- **Facial expressions** - Engagement indicators
- **Blink rate** - Attention monitoring

## License

This project uses various open-source tools. Please check individual licenses:
- Faster-Whisper: MIT
- PyAnnote: Apache 2.0
- SpeechBrain: Apache 2.0
- MediaPipe: Apache 2.0
- InsightFace: MIT
- Librosa: ISC

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Citation

If you use this system in your research, please cite the respective tools used.
