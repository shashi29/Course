"""
Voice Embeddings and Speaker Verification using SpeechBrain

SpeechBrain provides speaker recognition and verification capabilities,
enabling voice embeddings for speaker identification.

Link: https://github.com/speechbrain/speechbrain
"""

import numpy as np
from typing import List, Dict, Optional
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier


class VoiceEmbeddingExtractor:
    """
    Extracts voice embeddings for speaker verification using SpeechBrain.
    
    Features:
    - Speaker embedding extraction
    - Speaker verification (comparing two voice samples)
    - Speaker clustering
    - Pre-trained models available
    """
    
    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
                 device: str = "cpu"):
        """
        Initialize the speaker embedding extractor.
        
        Args:
            model_name: SpeechBrain model name for speaker recognition
            device: Device to run inference on ('cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading SpeechBrain speaker model: {model_name}...")
        self.classifier = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=f"/tmp/speechbrain/{model_name.split('/')[-1]}",
            run_opts={"device": device}
        )
        print("Speaker model loaded successfully!")
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract voice embedding from audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Numpy array containing the speaker embedding vector
        """
        embedding = self.classifier.encode_batch(audio_path)
        return embedding.squeeze().cpu().numpy()
    
    def extract_embedding_from_array(self, audio_array: np.ndarray, 
                                     sample_rate: int = 16000) -> np.ndarray:
        """
        Extract voice embedding from audio array.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Numpy array containing the speaker embedding vector
        """
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0).unsqueeze(0)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_tensor = resampler(audio_tensor)
        
        embedding = self.classifier.encode_batch(audio_tensor)
        return embedding.squeeze().cpu().numpy()
    
    def verify_speakers(self, audio_path_1: str, audio_path_2: str, 
                        threshold: float = 0.5) -> Dict:
        """
        Verify if two audio samples are from the same speaker.
        
        Args:
            audio_path_1: Path to first audio file
            audio_path_2: Path to second audio file
            threshold: Similarity threshold (default: 0.5)
            
        Returns:
            Dictionary containing:
                - is_same_speaker: Boolean result
                - similarity_score: Cosine similarity between embeddings
                - threshold: Used threshold
        """
        embedding1 = self.extract_embedding(audio_path_1)
        embedding2 = self.extract_embedding(audio_path_2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return {
            'is_same_speaker': similarity >= threshold,
            'similarity_score': float(similarity),
            'threshold': threshold
        }
    
    def cluster_speakers(self, audio_paths: List[str], 
                         num_speakers: int = None) -> Dict:
        """
        Cluster audio samples by speaker.
        
        Args:
            audio_paths: List of paths to audio files
            num_speakers: Expected number of speakers (optional)
            
        Returns:
            Dictionary mapping cluster IDs to audio file paths
        """
        from sklearn.cluster import KMeans
        
        # Extract embeddings for all audio files
        embeddings = []
        for path in audio_paths:
            emb = self.extract_embedding(path)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Determine number of clusters
        if num_speakers is None:
            # Use simple heuristic or could use more sophisticated methods
            num_speakers = min(2, len(audio_paths))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_speakers, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Group by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            cluster_key = f"speaker_{label}"
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append({
                'path': audio_paths[idx],
                'confidence': float(kmeans.transform(embeddings[idx:min(idx+1, len(embeddings))])[0][label])
            })
        
        return clusters
    
    def get_speaker_profile(self, audio_path: str) -> Dict:
        """
        Create a speaker profile from audio.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing speaker profile information
        """
        embedding = self.extract_embedding(audio_path)
        
        return {
            'embedding': embedding,
            'embedding_dimension': len(embedding),
            'embedding_norm': float(np.linalg.norm(embedding)),
            'audio_source': audio_path
        }


if __name__ == "__main__":
    # Example usage
    print("Voice Embedding Extractor Module")
    print("=" * 50)
    print("This module uses SpeechBrain for speaker verification.")
    print("\nInitialize with:")
    print("  extractor = VoiceEmbeddingExtractor()")
    print("\nExtract embedding with:")
    print("  embedding = extractor.extract_embedding('audio.wav')")
    print("\nVerify speakers with:")
    print("  result = extractor.verify_speakers('audio1.wav', 'audio2.wav')")
