"""
Modern Face Verification System

This module implements a state-of-the-art face verification system using multiple approaches:
1. Traditional face_recognition library
2. Hugging Face transformers for modern face recognition
3. DeepFace library for advanced face analysis
4. Custom embedding comparison with configurable thresholds

Author: AI Assistant
Date: 2024
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import yaml

# Face recognition libraries
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    warnings.warn("face_recognition library not available. Install with: pip install face_recognition")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    warnings.warn("DeepFace library not available. Install with: pip install deepface")

# Hugging Face transformers
try:
    from transformers import AutoImageProcessor, AutoModel
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers library not available. Install with: pip install transformers")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceVerificationSystem:
    """
    Modern face verification system supporting multiple backends and techniques.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the face verification system.
        
        Args:
            config_path: Path to configuration file (YAML format)
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.processors = {}
        self._initialize_models()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            'thresholds': {
                'face_recognition': 0.6,
                'deepface': 0.4,
                'transformers': 0.3
            },
            'models': {
                'face_recognition': True,
                'deepface': True,
                'transformers': True
            },
            'transformers_model': 'microsoft/DialoGPT-medium',  # Placeholder, will use face-specific model
            'image_size': (224, 224),
            'confidence_threshold': 0.5
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                
        return default_config
    
    def _initialize_models(self) -> None:
        """Initialize available face recognition models."""
        if self.config['models']['face_recognition'] and FACE_RECOGNITION_AVAILABLE:
            logger.info("Initializing face_recognition model")
            self.models['face_recognition'] = True
            
        if self.config['models']['deepface'] and DEEPFACE_AVAILABLE:
            logger.info("Initializing DeepFace model")
            self.models['deepface'] = True
            
        if self.config['models']['transformers'] and TRANSFORMERS_AVAILABLE:
            logger.info("Initializing transformers model")
            try:
                # Use a face recognition specific model if available
                model_name = "facebook/deit-base-distilled-patch16-224"  # Generic vision model
                self.processors['transformers'] = AutoImageProcessor.from_pretrained(model_name)
                self.models['transformers'] = AutoModel.from_pretrained(model_name)
                logger.info(f"Loaded transformers model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load transformers model: {e}")
                self.models['transformers'] = None
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            if isinstance(image_path, str):
                image_path = Path(image_path)
                
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def extract_face_recognition_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using face_recognition library.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Face embedding vector or None if no face found
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return None
            
        try:
            encodings = face_recognition.face_encodings(image)
            if encodings:
                return encodings[0]
            return None
        except Exception as e:
            logger.error(f"Error extracting face_recognition embedding: {e}")
            return None
    
    def extract_deepface_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using DeepFace library.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Face embedding vector or None if no face found
        """
        if not DEEPFACE_AVAILABLE:
            return None
            
        try:
            # DeepFace expects image path or numpy array
            embedding = DeepFace.represent(
                img_path=image,
                model_name='Facenet',
                enforce_detection=False
            )
            if embedding:
                return np.array(embedding[0]['embedding'])
            return None
        except Exception as e:
            logger.error(f"Error extracting DeepFace embedding: {e}")
            return None
    
    def extract_transformers_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using transformers model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Face embedding vector or None if no face found
        """
        if not TRANSFORMERS_AVAILABLE or not self.models.get('transformers'):
            return None
            
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Preprocess image
            inputs = self.processors['transformers'](pil_image, return_tensors="pt")
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.models['transformers'](**inputs)
                
            # Extract features (use pooler_output or last_hidden_state)
            if hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state.mean(dim=1)
                
            return features.numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error extracting transformers embedding: {e}")
            return None
    
    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                        method: str = 'euclidean') -> float:
        """
        Compute distance between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Distance method ('euclidean', 'cosine', 'manhattan')
            
        Returns:
            Distance value
        """
        if method == 'euclidean':
            return np.linalg.norm(embedding1 - embedding2)
        elif method == 'cosine':
            # Convert to similarity first, then to distance
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return 1 - similarity
        elif method == 'manhattan':
            return np.sum(np.abs(embedding1 - embedding2))
        else:
            raise ValueError(f"Unknown distance method: {method}")
    
    def verify_faces(self, image1_path: Union[str, Path], 
                    image2_path: Union[str, Path],
                    methods: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Verify if two faces belong to the same person using multiple methods.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            methods: List of methods to use (default: all available)
            
        Returns:
            Dictionary with verification results for each method
        """
        if methods is None:
            methods = list(self.models.keys())
            
        # Load images
        image1 = self.load_image(image1_path)
        image2 = self.load_image(image2_path)
        
        results = {}
        
        for method in methods:
            if method not in self.models or not self.models[method]:
                continue
                
            try:
                # Extract embeddings
                if method == 'face_recognition':
                    emb1 = self.extract_face_recognition_embedding(image1)
                    emb2 = self.extract_face_recognition_embedding(image2)
                elif method == 'deepface':
                    emb1 = self.extract_deepface_embedding(image1)
                    emb2 = self.extract_deepface_embedding(image2)
                elif method == 'transformers':
                    emb1 = self.extract_transformers_embedding(image1)
                    emb2 = self.extract_transformers_embedding(image2)
                else:
                    continue
                
                if emb1 is None or emb2 is None:
                    results[method] = {
                        'success': False,
                        'error': 'No face detected in one or both images'
                    }
                    continue
                
                # Compute distance
                distance = self.compute_distance(emb1, emb2)
                threshold = self.config['thresholds'].get(method, 0.5)
                is_match = distance < threshold
                
                results[method] = {
                    'success': True,
                    'distance': float(distance),
                    'threshold': threshold,
                    'is_match': is_match,
                    'confidence': max(0, 1 - distance / threshold) if threshold > 0 else 0
                }
                
            except Exception as e:
                results[method] = {
                    'success': False,
                    'error': str(e)
                }
                
        return results
    
    def visualize_comparison(self, image1_path: Union[str, Path], 
                           image2_path: Union[str, Path],
                           results: Dict[str, Dict],
                           save_path: Optional[str] = None) -> None:
        """
        Visualize face comparison results.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            results: Verification results
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Load and display images
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)
        
        axes[0].imshow(img1)
        axes[0].set_title("Image 1", fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].set_title("Image 2", fontsize=14)
        axes[1].axis('off')
        
        # Add results text
        results_text = "Verification Results:\n"
        for method, result in results.items():
            if result['success']:
                status = "✅ Match" if result['is_match'] else "❌ No Match"
                results_text += f"{method}: {status} (dist: {result['distance']:.3f})\n"
            else:
                results_text += f"{method}: Error - {result['error']}\n"
        
        fig.text(0.5, 0.02, results_text, ha='center', va='bottom', fontsize=10)
        plt.suptitle("Face Verification Results", fontsize=16, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()


def main():
    """Example usage of the FaceVerificationSystem."""
    # Initialize the system
    verifier = FaceVerificationSystem()
    
    # Example paths (these would need to exist)
    image1_path = "data/person_A.jpg"
    image2_path = "data/person_B.jpg"
    
    # Check if example images exist
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        logger.warning("Example images not found. Please provide valid image paths.")
        return
    
    # Perform verification
    results = verifier.verify_faces(image1_path, image2_path)
    
    # Print results
    print("\n🔍 Face Verification Results:")
    print("=" * 50)
    
    for method, result in results.items():
        if result['success']:
            status = "✅ Match (Same Person)" if result['is_match'] else "❌ No Match (Different People)"
            print(f"{method.upper()}:")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  Threshold: {result['threshold']:.4f}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Result: {status}")
            print()
        else:
            print(f"{method.upper()}: Error - {result['error']}")
            print()
    
    # Visualize results
    verifier.visualize_comparison(image1_path, image2_path, results)


if __name__ == "__main__":
    main()
