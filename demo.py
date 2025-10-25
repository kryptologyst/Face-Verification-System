#!/usr/bin/env python3
"""
Demo Script for Face Verification System

This script demonstrates the capabilities of the face verification system
by running various examples and showcasing different features.
"""

import logging
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from face_verification import FaceVerificationSystem
from data_generator import create_sample_dataset, MockDatabase, visualize_sample_images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_verification():
    """Demonstrate basic face verification functionality."""
    print("\n" + "="*60)
    print("🔍 DEMO: Basic Face Verification")
    print("="*60)
    
    try:
        # Initialize the system
        config_path = Path(__file__).parent / "config" / "config.yaml"
        verifier = FaceVerificationSystem(str(config_path))
        
        print("✅ Face verification system initialized successfully")
        print(f"📋 Available methods: {list(verifier.models.keys())}")
        
        # Create sample data if it doesn't exist
        data_dir = Path(__file__).parent / "data"
        if not (data_dir / "mock_database.json").exists():
            print("\n🎲 Creating sample dataset...")
            create_sample_dataset(str(data_dir), num_persons=5)
            print("✅ Sample dataset created")
        
        # Load database and get sample images
        database = MockDatabase(str(data_dir / "mock_database.json"))
        person_ids = list(database.data['persons'].keys())
        
        if len(person_ids) >= 2:
            # Test same person verification
            person1_id = person_ids[0]
            person1_data = database.get_person(person1_id)
            
            if person1_data and len(person1_data['image_paths']) >= 2:
                img1_path = person1_data['image_paths'][0]
                img2_path = person1_data['image_paths'][1]
                
                print(f"\n🔍 Testing same person verification: {person1_id}")
                print(f"   Image 1: {img1_path}")
                print(f"   Image 2: {img2_path}")
                
                results = verifier.verify_faces(img1_path, img2_path)
                
                print("\n📊 Results:")
                for method, result in results.items():
                    if result['success']:
                        status = "✅ Match" if result['is_match'] else "❌ No Match"
                        print(f"   {method}: {status} (distance: {result['distance']:.4f})")
                    else:
                        print(f"   {method}: Error - {result['error']}")
            
            # Test different person verification
            if len(person_ids) >= 2:
                person2_id = person_ids[1]
                person2_data = database.get_person(person2_id)
                
                if person1_data and person2_data and person1_data['image_paths'] and person2_data['image_paths']:
                    img1_path = person1_data['image_paths'][0]
                    img2_path = person2_data['image_paths'][0]
                    
                    print(f"\n🔍 Testing different person verification: {person1_id} vs {person2_id}")
                    print(f"   Image 1: {img1_path}")
                    print(f"   Image 2: {img2_path}")
                    
                    results = verifier.verify_faces(img1_path, img2_path)
                    
                    print("\n📊 Results:")
                    for method, result in results.items():
                        if result['success']:
                            status = "✅ Match" if result['is_match'] else "❌ No Match"
                            print(f"   {method}: {status} (distance: {result['distance']:.4f})")
                        else:
                            print(f"   {method}: Error - {result['error']}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo error: {e}")


def demo_distance_methods():
    """Demonstrate different distance calculation methods."""
    print("\n" + "="*60)
    print("📏 DEMO: Distance Calculation Methods")
    print("="*60)
    
    try:
        # Create sample embeddings
        import numpy as np
        
        emb1 = np.array([1, 2, 3, 4, 5])
        emb2 = np.array([2, 3, 4, 5, 6])
        
        verifier = FaceVerificationSystem()
        
        methods = ['euclidean', 'cosine', 'manhattan']
        
        print(f"Embedding 1: {emb1}")
        print(f"Embedding 2: {emb2}")
        print()
        
        for method in methods:
            distance = verifier.compute_distance(emb1, emb2, method)
            print(f"{method.capitalize()} distance: {distance:.4f}")
        
    except Exception as e:
        logger.error(f"Distance demo failed: {e}")
        print(f"❌ Distance demo error: {e}")


def demo_database_features():
    """Demonstrate database features."""
    print("\n" + "="*60)
    print("🗄️ DEMO: Database Features")
    print("="*60)
    
    try:
        data_dir = Path(__file__).parent / "data"
        database = MockDatabase(str(data_dir / "mock_database.json"))
        
        # Show database statistics
        persons_count = len(database.data['persons'])
        verifications_count = len(database.data['verification_history'])
        
        print(f"📊 Database Statistics:")
        print(f"   - Total persons: {persons_count}")
        print(f"   - Total verifications: {verifications_count}")
        
        if persons_count > 0:
            print(f"\n👥 Sample persons:")
            person_ids = list(database.data['persons'].keys())[:3]
            for person_id in person_ids:
                person_data = database.get_person(person_id)
                if person_data:
                    print(f"   - {person_id}: {len(person_data['image_paths'])} images")
        
        if verifications_count > 0:
            print(f"\n🔍 Recent verifications:")
            recent = database.get_verification_history(limit=3)
            for verification in recent:
                status = "✅ Match" if verification['is_match'] else "❌ No Match"
                print(f"   - {verification['person1_id']} vs {verification['person2_id']}: {status}")
        
    except Exception as e:
        logger.error(f"Database demo failed: {e}")
        print(f"❌ Database demo error: {e}")


def demo_configuration():
    """Demonstrate configuration features."""
    print("\n" + "="*60)
    print("⚙️ DEMO: Configuration Features")
    print("="*60)
    
    try:
        config_path = Path(__file__).parent / "config" / "config.yaml"
        verifier = FaceVerificationSystem(str(config_path))
        
        print("📋 Current Configuration:")
        print(f"   - Thresholds: {verifier.config['thresholds']}")
        print(f"   - Enabled models: {verifier.config['models']}")
        print(f"   - Image size: {verifier.config['image_size']}")
        print(f"   - Confidence threshold: {verifier.config['confidence_threshold']}")
        
    except Exception as e:
        logger.error(f"Configuration demo failed: {e}")
        print(f"❌ Configuration demo error: {e}")


def demo_visualization():
    """Demonstrate visualization features."""
    print("\n" + "="*60)
    print("📊 DEMO: Visualization Features")
    print("="*60)
    
    try:
        data_dir = Path(__file__).parent / "data"
        
        print("🎨 Generating visualization of sample images...")
        print("   (This will open a matplotlib window)")
        
        # Note: This will open a matplotlib window
        visualize_sample_images(str(data_dir), num_samples=4)
        
        print("✅ Visualization completed")
        
    except Exception as e:
        logger.error(f"Visualization demo failed: {e}")
        print(f"❌ Visualization demo error: {e}")


def main():
    """Main demo function."""
    print("🚀 FACE VERIFICATION SYSTEM DEMO")
    print("="*60)
    print("This demo showcases the capabilities of the face verification system.")
    print("Make sure you have installed all required dependencies.")
    
    # Run demos
    demo_configuration()
    demo_basic_verification()
    demo_distance_methods()
    demo_database_features()
    
    # Ask user if they want to see visualization
    try:
        response = input("\n🎨 Would you like to see the visualization demo? (y/n): ").lower()
        if response in ['y', 'yes']:
            demo_visualization()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print("1. Run the web interface: streamlit run web_app/app.py")
    print("2. Use the CLI: python cli.py verify image1.jpg image2.jpg")
    print("3. Run tests: python -m pytest tests/")
    print("4. Check the README.md for more information")


if __name__ == "__main__":
    main()
