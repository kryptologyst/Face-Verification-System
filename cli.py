#!/usr/bin/env python3
"""
Command Line Interface for Face Verification System

This module provides a command-line interface for the face verification system,
allowing users to perform face verification from the terminal.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from face_verification import FaceVerificationSystem
from data_generator import create_sample_dataset, visualize_sample_images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_faces_cli(image1_path: str, image2_path: str, 
                    config_path: Optional[str] = None,
                    methods: Optional[list] = None,
                    output_file: Optional[str] = None) -> None:
    """
    Perform face verification via command line.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        config_path: Path to configuration file
        methods: List of methods to use
        output_file: Optional output file for results
    """
    try:
        # Initialize verification system
        verifier = FaceVerificationSystem(config_path)
        
        # Perform verification
        logger.info(f"Verifying faces: {image1_path} vs {image2_path}")
        results = verifier.verify_faces(image1_path, image2_path, methods)
        
        # Display results
        print("\n" + "="*60)
        print("🔍 FACE VERIFICATION RESULTS")
        print("="*60)
        
        for method, result in results.items():
            print(f"\n{method.upper()}:")
            print("-" * 30)
            
            if result['success']:
                status = "✅ MATCH (Same Person)" if result['is_match'] else "❌ NO MATCH (Different People)"
                print(f"Result: {status}")
                print(f"Distance: {result['distance']:.4f}")
                print(f"Threshold: {result['threshold']:.4f}")
                print(f"Confidence: {result['confidence']:.2%}")
            else:
                print(f"Error: {result['error']}")
        
        # Overall result
        successful_results = [r for r in results.values() if r['success']]
        if successful_results:
            matches = sum(1 for r in successful_results if r['is_match'])
            total = len(successful_results)
            print(f"\n📊 OVERALL RESULT: {matches}/{total} methods indicate match")
        
        # Save results to file if requested
        if output_file:
            save_results_to_file(results, output_file)
            print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


def save_results_to_file(results: dict, output_file: str) -> None:
    """
    Save verification results to a file.
    
    Args:
        results: Verification results dictionary
        output_file: Output file path
    """
    import json
    from datetime import datetime
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'total_methods': len(results),
            'successful_methods': len([r for r in results.values() if r['success']]),
            'matches': len([r for r in results.values() if r['success'] and r['is_match']])
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)


def create_sample_data_cli(output_dir: str, num_persons: int) -> None:
    """
    Create sample dataset via command line.
    
    Args:
        output_dir: Output directory for the dataset
        num_persons: Number of persons to generate
    """
    try:
        logger.info(f"Creating sample dataset with {num_persons} persons...")
        create_sample_dataset(output_dir, num_persons)
        print(f"✅ Sample dataset created successfully in {output_dir}")
        
        # Show dataset statistics
        from data_generator import MockDatabase
        db_path = Path(output_dir) / "mock_database.json"
        if db_path.exists():
            database = MockDatabase(str(db_path))
            persons_count = len(database.data['persons'])
            verifications_count = len(database.data['verification_history'])
            
            print(f"📊 Dataset Statistics:")
            print(f"   - Persons: {persons_count}")
            print(f"   - Verifications: {verifications_count}")
            print(f"   - Images: {sum(len(p['image_paths']) for p in database.data['persons'].values())}")
        
    except Exception as e:
        logger.error(f"Failed to create sample dataset: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


def visualize_samples_cli(data_dir: str, num_samples: int) -> None:
    """
    Visualize sample images via command line.
    
    Args:
        data_dir: Directory containing the dataset
        num_samples: Number of samples to display
    """
    try:
        logger.info(f"Visualizing {num_samples} sample images...")
        visualize_sample_images(data_dir, num_samples)
        print("✅ Visualization completed")
        
    except Exception as e:
        logger.error(f"Failed to visualize samples: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Face Verification System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify two faces
  python cli.py verify image1.jpg image2.jpg
  
  # Verify with specific methods
  python cli.py verify image1.jpg image2.jpg --methods face_recognition deepface
  
  # Verify and save results
  python cli.py verify image1.jpg image2.jpg --output results.json
  
  # Create sample dataset
  python cli.py create-sample --output-dir data --num-persons 10
  
  # Visualize samples
  python cli.py visualize --data-dir data --num-samples 6
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify two face images')
    verify_parser.add_argument('image1', help='Path to first image')
    verify_parser.add_argument('image2', help='Path to second image')
    verify_parser.add_argument('--config', help='Path to configuration file')
    verify_parser.add_argument('--methods', nargs='+', 
                             choices=['face_recognition', 'deepface', 'transformers'],
                             help='Verification methods to use')
    verify_parser.add_argument('--output', help='Output file for results')
    
    # Create sample command
    create_parser = subparsers.add_parser('create-sample', help='Create sample dataset')
    create_parser.add_argument('--output-dir', default='data', 
                             help='Output directory for dataset')
    create_parser.add_argument('--num-persons', type=int, default=10,
                             help='Number of persons to generate')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize sample images')
    viz_parser.add_argument('--data-dir', default='data',
                           help='Directory containing the dataset')
    viz_parser.add_argument('--num-samples', type=int, default=6,
                           help='Number of samples to display')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute commands
    if args.command == 'verify':
        verify_faces_cli(
            args.image1, 
            args.image2, 
            args.config, 
            args.methods, 
            args.output
        )
    elif args.command == 'create-sample':
        create_sample_data_cli(args.output_dir, args.num_persons)
    elif args.command == 'visualize':
        visualize_samples_cli(args.data_dir, args.num_samples)


if __name__ == "__main__":
    main()
