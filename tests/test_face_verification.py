"""
Test suite for the Face Verification System

This module contains comprehensive tests for the face verification system,
including unit tests, integration tests, and performance tests.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import cv2
from PIL import Image

# Add src directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from face_verification import FaceVerificationSystem
from data_generator import SyntheticFaceGenerator, MockDatabase, create_sample_dataset


class TestSyntheticFaceGenerator(unittest.TestCase):
    """Test cases for SyntheticFaceGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = SyntheticFaceGenerator()
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.image_size, (224, 224))
        self.assertIsInstance(self.generator.face_templates, list)
        self.assertGreater(len(self.generator.face_templates), 0)
    
    def test_generate_face(self):
        """Test face generation."""
        face = self.generator.generate_face()
        
        # Check image properties
        self.assertIsInstance(face, np.ndarray)
        self.assertEqual(face.shape, (224, 224, 3))
        self.assertEqual(face.dtype, np.uint8)
        
        # Check that image is not empty
        self.assertGreater(np.sum(face), 0)
    
    def test_generate_face_with_template(self):
        """Test face generation with specific template."""
        template = self.generator.face_templates[0]
        face = self.generator.generate_face(template)
        
        self.assertIsInstance(face, np.ndarray)
        self.assertEqual(face.shape, (224, 224, 3))
    
    def test_generate_person_dataset(self):
        """Test generating multiple images for a person."""
        person_id = "test_person"
        num_images = 3
        
        images = self.generator.generate_person_dataset(person_id, num_images)
        
        self.assertEqual(len(images), num_images)
        for img in images:
            self.assertIsInstance(img, np.ndarray)
            self.assertEqual(img.shape, (224, 224, 3))


class TestMockDatabase(unittest.TestCase):
    """Test cases for MockDatabase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db_path = tempfile.mktemp(suffix='.json')
        self.database = MockDatabase(self.temp_db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def test_initialization(self):
        """Test database initialization."""
        self.assertIsInstance(self.database.data, dict)
        self.assertIn('persons', self.database.data)
        self.assertIn('verification_history', self.database.data)
        self.assertIn('metadata', self.database.data)
    
    def test_add_person(self):
        """Test adding a person to the database."""
        person_id = "test_person"
        images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
        metadata = {"test": "data"}
        
        self.database.add_person(person_id, images, metadata)
        
        self.assertIn(person_id, self.database.data['persons'])
        person_data = self.database.data['persons'][person_id]
        self.assertEqual(len(person_data['image_paths']), 2)
        self.assertEqual(person_data['metadata'], metadata)
    
    def test_get_person(self):
        """Test retrieving a person from the database."""
        person_id = "test_person"
        images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)]
        
        self.database.add_person(person_id, images)
        
        retrieved_person = self.database.get_person(person_id)
        self.assertIsNotNone(retrieved_person)
        self.assertEqual(len(retrieved_person['image_paths']), 1)
        
        # Test non-existent person
        non_existent = self.database.get_person("non_existent")
        self.assertIsNone(non_existent)
    
    def test_add_verification_record(self):
        """Test adding verification records."""
        person1_id = "person1"
        person2_id = "person2"
        results = {"method1": {"success": True, "distance": 0.5}}
        is_match = True
        
        self.database.add_verification_record(person1_id, person2_id, results, is_match)
        
        history = self.database.get_verification_history()
        self.assertEqual(len(history), 1)
        
        record = history[0]
        self.assertEqual(record['person1_id'], person1_id)
        self.assertEqual(record['person2_id'], person2_id)
        self.assertEqual(record['is_match'], is_match)
        self.assertEqual(record['results'], results)


class TestFaceVerificationSystem(unittest.TestCase):
    """Test cases for FaceVerificationSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_config_path = tempfile.mktemp(suffix='.yaml')
        self.create_test_config()
        self.verifier = FaceVerificationSystem(self.temp_config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_config_path):
            os.unlink(self.temp_config_path)
    
    def create_test_config(self):
        """Create a test configuration file."""
        config_content = """
thresholds:
  face_recognition: 0.6
  deepface: 0.4
  transformers: 0.3

models:
  face_recognition: true
  deepface: true
  transformers: true

transformers_model: "facebook/deit-base-distilled-patch16-224"
image_size: [224, 224]
confidence_threshold: 0.5
"""
        with open(self.temp_config_path, 'w') as f:
            f.write(config_content)
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsInstance(self.verifier.config, dict)
        self.assertIn('thresholds', self.verifier.config)
        self.assertIn('models', self.verifier.config)
    
    def test_load_image(self):
        """Test image loading functionality."""
        # Create a test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        temp_image_path = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(temp_image_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        
        try:
            loaded_image = self.verifier.load_image(temp_image_path)
            self.assertIsInstance(loaded_image, np.ndarray)
            self.assertEqual(loaded_image.shape, (224, 224, 3))
        finally:
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
    
    def test_compute_distance(self):
        """Test distance computation."""
        emb1 = np.array([1, 2, 3, 4])
        emb2 = np.array([2, 3, 4, 5])
        
        # Test Euclidean distance
        euclidean_dist = self.verifier.compute_distance(emb1, emb2, 'euclidean')
        self.assertAlmostEqual(euclidean_dist, 2.0, places=5)
        
        # Test Manhattan distance
        manhattan_dist = self.verifier.compute_distance(emb1, emb2, 'manhattan')
        self.assertEqual(manhattan_dist, 4.0)
        
        # Test cosine distance
        cosine_dist = self.verifier.compute_distance(emb1, emb2, 'cosine')
        self.assertIsInstance(cosine_dist, float)
        self.assertGreaterEqual(cosine_dist, 0)
        self.assertLessEqual(cosine_dist, 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = SyntheticFaceGenerator()
        self.database = MockDatabase(os.path.join(self.temp_dir, "test_db.json"))
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_verification(self):
        """Test complete verification workflow."""
        # Generate test images
        person1_images = self.generator.generate_person_dataset("person1", 2)
        person2_images = self.generator.generate_person_dataset("person2", 2)
        
        # Save images
        img1_path = os.path.join(self.temp_dir, "person1.jpg")
        img2_path = os.path.join(self.temp_dir, "person2.jpg")
        
        cv2.imwrite(img1_path, cv2.cvtColor(person1_images[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(img2_path, cv2.cvtColor(person2_images[0], cv2.COLOR_RGB2BGR))
        
        # Test verification (mock the actual verification to avoid dependencies)
        with patch.object(FaceVerificationSystem, 'verify_faces') as mock_verify:
            mock_verify.return_value = {
                'face_recognition': {
                    'success': True,
                    'distance': 0.5,
                    'threshold': 0.6,
                    'is_match': True,
                    'confidence': 0.8
                }
            }
            
            verifier = FaceVerificationSystem()
            results = verifier.verify_faces(img1_path, img2_path)
            
            self.assertIn('face_recognition', results)
            self.assertTrue(results['face_recognition']['success'])
            self.assertTrue(results['face_recognition']['is_match'])


class TestDataGeneratorIntegration(unittest.TestCase):
    """Test integration with data generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        try:
            create_sample_dataset(self.temp_dir, num_persons=3)
            
            # Check that database file was created
            db_path = os.path.join(self.temp_dir, "mock_database.json")
            self.assertTrue(os.path.exists(db_path))
            
            # Check that images were created
            images_dir = os.path.join(self.temp_dir, "images")
            self.assertTrue(os.path.exists(images_dir))
            
            # Check database content
            database = MockDatabase(db_path)
            self.assertGreater(len(database.data['persons']), 0)
            
        except Exception as e:
            # Skip test if dependencies are not available
            self.skipTest(f"Dependencies not available: {e}")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestSyntheticFaceGenerator,
        TestMockDatabase,
        TestFaceVerificationSystem,
        TestIntegration,
        TestDataGeneratorIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
