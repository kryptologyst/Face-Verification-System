"""
Synthetic Dataset Generator for Face Verification System

This module generates synthetic face images and creates a mock database
for testing the face verification system without requiring real face images.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SyntheticFaceGenerator:
    """
    Generates synthetic face-like images for testing purposes.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the synthetic face generator.
        
        Args:
            image_size: Size of generated images (width, height)
        """
        self.image_size = image_size
        self.face_templates = self._create_face_templates()
        
    def _create_face_templates(self) -> List[Dict]:
        """Create basic face templates with different characteristics."""
        templates = []
        
        # Template 1: Round face
        templates.append({
            'name': 'round_face',
            'face_shape': 'round',
            'eye_color': (50, 50, 150),
            'skin_color': (220, 180, 140),
            'hair_color': (80, 50, 30)
        })
        
        # Template 2: Oval face
        templates.append({
            'name': 'oval_face',
            'face_shape': 'oval',
            'eye_color': (30, 100, 30),
            'skin_color': (200, 160, 120),
            'hair_color': (60, 40, 20)
        })
        
        # Template 3: Square face
        templates.append({
            'name': 'square_face',
            'face_shape': 'square',
            'eye_color': (150, 50, 50),
            'skin_color': (240, 200, 160),
            'hair_color': (100, 60, 40)
        })
        
        return templates
    
    def generate_face(self, template: Optional[Dict] = None, 
                     variation: float = 0.1) -> np.ndarray:
        """
        Generate a synthetic face image.
        
        Args:
            template: Face template to use (random if None)
            variation: Amount of random variation to add
            
        Returns:
            Generated face image as numpy array
        """
        if template is None:
            template = random.choice(self.face_templates)
            
        # Create base image
        img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # Add background
        img.fill(240)
        
        # Draw face shape
        face_center = (self.image_size[0] // 2, self.image_size[1] // 2)
        face_size = min(self.image_size) // 3
        
        # Apply variation
        face_size = int(face_size * (1 + random.uniform(-variation, variation)))
        
        # Draw face oval
        cv2.ellipse(img, face_center, (face_size, int(face_size * 1.2)), 
                    template['skin_color'], -1)
        
        # Draw eyes
        eye_y = face_center[1] - face_size // 3
        left_eye = (face_center[0] - face_size // 3, eye_y)
        right_eye = (face_center[0] + face_size // 3, eye_y)
        
        cv2.circle(img, left_eye, face_size // 8, template['eye_color'], -1)
        cv2.circle(img, right_eye, face_size // 8, template['eye_color'], -1)
        
        # Draw pupils
        cv2.circle(img, left_eye, face_size // 12, (0, 0, 0), -1)
        cv2.circle(img, right_eye, face_size // 12, (0, 0, 0), -1)
        
        # Draw nose
        nose_y = face_center[1]
        cv2.ellipse(img, (face_center[0], nose_y), 
                   (face_size // 12, face_size // 8), 
                   template['skin_color'], -1)
        
        # Draw mouth
        mouth_y = face_center[1] + face_size // 3
        cv2.ellipse(img, (face_center[0], mouth_y), 
                   (face_size // 4, face_size // 8), 
                   (150, 50, 50), -1)
        
        # Draw hair
        hair_y = face_center[1] - face_size
        cv2.ellipse(img, (face_center[0], hair_y), 
                   (face_size, face_size // 2), 
                   template['hair_color'], -1)
        
        return img
    
    def generate_person_dataset(self, person_id: str, num_images: int = 5) -> List[np.ndarray]:
        """
        Generate multiple images of the same synthetic person.
        
        Args:
            person_id: Unique identifier for the person
            num_images: Number of images to generate
            
        Returns:
            List of generated face images
        """
        # Choose a template for this person
        template = random.choice(self.face_templates)
        
        images = []
        for i in range(num_images):
            # Add slight variations to make images look different
            variation = random.uniform(0.05, 0.15)
            img = self.generate_face(template, variation)
            images.append(img)
            
        return images


class MockDatabase:
    """
    Mock database for storing face verification data.
    """
    
    def __init__(self, db_path: str = "data/mock_database.json"):
        """
        Initialize the mock database.
        
        Args:
            db_path: Path to the database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load_database()
        
    def _load_database(self) -> Dict:
        """Load database from file or create new one."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load database: {e}")
                
        return {
            'persons': {},
            'verification_history': [],
            'metadata': {
                'created_at': str(uuid.uuid4()),
                'version': '1.0'
            }
        }
    
    def save_database(self) -> None:
        """Save database to file."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"Database saved to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def add_person(self, person_id: str, images: List[np.ndarray], 
                  metadata: Optional[Dict] = None) -> None:
        """
        Add a person to the database.
        
        Args:
            person_id: Unique identifier for the person
            images: List of face images
            metadata: Additional metadata about the person
        """
        if metadata is None:
            metadata = {}
            
        # Convert images to base64 for storage (simplified - in real app, use proper storage)
        image_paths = []
        for i, img in enumerate(images):
            img_path = f"data/images/{person_id}_{i}.jpg"
            Path(img_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            image_paths.append(img_path)
        
        self.data['persons'][person_id] = {
            'image_paths': image_paths,
            'metadata': metadata,
            'created_at': str(uuid.uuid4())
        }
        
        self.save_database()
    
    def get_person(self, person_id: str) -> Optional[Dict]:
        """
        Get person data from database.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Person data or None if not found
        """
        return self.data['persons'].get(person_id)
    
    def add_verification_record(self, person1_id: str, person2_id: str, 
                              results: Dict, is_match: bool) -> None:
        """
        Add a verification record to the database.
        
        Args:
            person1_id: First person identifier
            person2_id: Second person identifier
            results: Verification results
            is_match: Whether the verification was successful
        """
        record = {
            'id': str(uuid.uuid4()),
            'person1_id': person1_id,
            'person2_id': person2_id,
            'results': results,
            'is_match': is_match,
            'timestamp': str(uuid.uuid4())  # In real app, use proper timestamp
        }
        
        self.data['verification_history'].append(record)
        self.save_database()
    
    def get_verification_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get verification history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of verification records
        """
        history = self.data['verification_history']
        if limit:
            return history[-limit:]
        return history


def create_sample_dataset(output_dir: str = "data", num_persons: int = 10) -> None:
    """
    Create a sample dataset for testing.
    
    Args:
        output_dir: Output directory for the dataset
        num_persons: Number of persons to generate
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticFaceGenerator()
    database = MockDatabase(str(output_path / "mock_database.json"))
    
    logger.info(f"Creating sample dataset with {num_persons} persons...")
    
    for i in range(num_persons):
        person_id = f"person_{i:03d}"
        
        # Generate 3-5 images per person
        num_images = random.randint(3, 5)
        images = generator.generate_person_dataset(person_id, num_images)
        
        # Add to database
        metadata = {
            'generated': True,
            'template': generator.face_templates[i % len(generator.face_templates)]['name']
        }
        database.add_person(person_id, images, metadata)
        
        logger.info(f"Generated person {person_id} with {num_images} images")
    
    logger.info(f"Sample dataset created in {output_dir}")
    
    # Create some example verification pairs
    person_ids = list(database.data['persons'].keys())
    
    # Add some same-person verifications
    for i in range(5):
        person_id = random.choice(person_ids)
        person_data = database.get_person(person_id)
        if person_data and len(person_data['image_paths']) >= 2:
            img1, img2 = random.sample(person_data['image_paths'], 2)
            # Mock verification result
            mock_results = {
                'face_recognition': {'success': True, 'distance': 0.3, 'is_match': True},
                'deepface': {'success': True, 'distance': 0.2, 'is_match': True}
            }
            database.add_verification_record(person_id, person_id, mock_results, True)
    
    # Add some different-person verifications
    for i in range(5):
        person1_id, person2_id = random.sample(person_ids, 2)
        mock_results = {
            'face_recognition': {'success': True, 'distance': 0.8, 'is_match': False},
            'deepface': {'success': True, 'distance': 0.7, 'is_match': False}
        }
        database.add_verification_record(person1_id, person2_id, mock_results, False)


def visualize_sample_images(output_dir: str = "data", num_samples: int = 6) -> None:
    """
    Visualize sample images from the generated dataset.
    
    Args:
        output_dir: Directory containing the dataset
        num_samples: Number of sample images to display
    """
    database = MockDatabase(str(Path(output_dir) / "mock_database.json"))
    
    if not database.data['persons']:
        logger.warning("No persons in database to visualize")
        return
    
    # Get sample persons
    person_ids = list(database.data['persons'].keys())[:num_samples]
    
    fig, axes = plt.subplots(2, len(person_ids), figsize=(15, 6))
    if len(person_ids) == 1:
        axes = axes.reshape(2, 1)
    
    for i, person_id in enumerate(person_ids):
        person_data = database.get_person(person_id)
        if person_data and person_data['image_paths']:
            # Load first image
            img1_path = person_data['image_paths'][0]
            img1 = cv2.imread(img1_path)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            
            axes[0, i].imshow(img1)
            axes[0, i].set_title(f"{person_id} - Image 1")
            axes[0, i].axis('off')
            
            # Load second image if available
            if len(person_data['image_paths']) > 1:
                img2_path = person_data['image_paths'][1]
                img2 = cv2.imread(img2_path)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                
                axes[1, i].imshow(img2)
                axes[1, i].set_title(f"{person_id} - Image 2")
            else:
                axes[1, i].text(0.5, 0.5, 'No second image', 
                              ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f"{person_id} - No Image 2")
            axes[1, i].axis('off')
    
    plt.suptitle("Sample Generated Face Images", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create sample dataset
    create_sample_dataset()
    
    # Visualize samples
    visualize_sample_images()
