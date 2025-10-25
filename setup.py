#!/usr/bin/env python3
"""
Setup script for Face Verification System

This script helps set up the face verification system by installing dependencies,
creating necessary directories, and running initial setup tasks.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        command: Command to run
        description: Description of what the command does
        
    Returns:
        True if command succeeded, False otherwise
    """
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "data/images", 
        "models",
        "logs",
        "tests",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")


def create_sample_data():
    """Create sample data for testing."""
    print("🎲 Creating sample data...")
    
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from data_generator import create_sample_dataset
        
        create_sample_dataset("data", num_persons=5)
        print("✅ Sample data created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    if not Path("tests").exists():
        print("⚠️ Tests directory not found, skipping tests")
        return True
    
    return run_command("python -m pytest tests/ -v", "Running test suite")


def main():
    """Main setup function."""
    print("🚀 FACE VERIFICATION SYSTEM SETUP")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create sample data
    if not create_sample_data():
        print("⚠️ Failed to create sample data (this is optional)")
    
    # Run tests
    if not run_tests():
        print("⚠️ Tests failed (this might be due to missing dependencies)")
    
    print("\n" + "="*50)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    print("\n🎉 Your face verification system is ready!")
    print("\nNext steps:")
    print("1. Run the web interface: streamlit run web_app/app.py")
    print("2. Try the CLI: python cli.py verify --help")
    print("3. Run the demo: python demo.py")
    print("4. Check README.md for more information")
    
    print("\n📚 Available commands:")
    print("   - Web interface: streamlit run web_app/app.py")
    print("   - CLI verification: python cli.py verify image1.jpg image2.jpg")
    print("   - Create sample data: python cli.py create-sample")
    print("   - Run demo: python demo.py")
    print("   - Run tests: python -m pytest tests/")


if __name__ == "__main__":
    main()
