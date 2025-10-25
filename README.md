# Face Verification System

State-of-the-art face verification system that determines whether two face images belong to the same person or different people. This project implements multiple face recognition backends with a user-friendly web interface.

## Features

- **Multiple Backends**: Supports face_recognition, DeepFace, and Hugging Face transformers
- **Configurable Thresholds**: Adjustable verification thresholds for each method
- **Multiple Distance Metrics**: Euclidean, cosine, and Manhattan distance calculations
- **Interactive Web Interface**: Easy-to-use Streamlit interface
- **Sample Data Generation**: Synthetic face generation for testing
- **Database Integration**: Mock database for storing verification history
- **Visualization**: Interactive charts and result analysis
- **Modern Architecture**: Clean code structure with type hints and comprehensive documentation

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face-verification-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the web application**
   ```bash
   streamlit run web_app/app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
face-verification-system/
├── src/                    # Source code
│   ├── face_verification.py    # Core face verification system
│   └── data_generator.py      # Synthetic data generation
├── web_app/               # Web interface
│   └── app.py                 # Streamlit application
├── config/                # Configuration files
│   └── config.yaml            # System configuration
├── data/                  # Data storage
│   ├── images/                # Generated images
│   └── mock_database.json     # Mock database
├── tests/                 # Test files
├── models/                # Model storage
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🔧 Usage

### Web Interface

1. **Upload Images**: Use the "Upload Images" tab to upload two face images
2. **Select Methods**: Choose which verification methods to use
3. **Verify**: Click "Verify Faces" to get results
4. **View Results**: See detailed analysis and visualizations

### Sample Data

1. **Generate Sample Data**: Click "Generate Sample Dataset" in the Sample Data tab
2. **Select Persons**: Choose two persons from the dropdown menus
3. **Verify**: Click "Verify Sample Faces" to test the system

### Programmatic Usage

```python
from src.face_verification import FaceVerificationSystem

# Initialize the system
verifier = FaceVerificationSystem("config/config.yaml")

# Verify two faces
results = verifier.verify_faces("path/to/image1.jpg", "path/to/image2.jpg")

# Print results
for method, result in results.items():
    if result['success']:
        print(f"{method}: {'Match' if result['is_match'] else 'No Match'}")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Confidence: {result['confidence']:.2%}")
```

## Supported Methods

### 1. Face Recognition Library
- **Library**: `face_recognition`
- **Model**: HOG-based face detection + CNN-based face encoding
- **Features**: 128-dimensional face embeddings
- **Best for**: General-purpose face verification

### 2. DeepFace
- **Library**: `deepface`
- **Model**: Facenet (default)
- **Features**: Multiple model support (VGG-Face, Facenet, OpenFace, etc.)
- **Best for**: High-accuracy face verification

### 3. Hugging Face Transformers
- **Library**: `transformers`
- **Model**: Vision Transformer models
- **Features**: Modern transformer-based face analysis
- **Best for**: Research and cutting-edge applications

## Configuration

The system can be configured via `config/config.yaml`:

```yaml
# Distance thresholds for different methods
thresholds:
  face_recognition: 0.6
  deepface: 0.4
  transformers: 0.3

# Enable/disable different methods
models:
  face_recognition: true
  deepface: true
  transformers: true

# Distance methods
distance_methods:
  - euclidean
  - cosine
  - manhattan
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_face_verification.py
```

## Performance

The system provides detailed performance metrics:

- **Distance Scores**: Raw similarity distances
- **Confidence Levels**: Normalized confidence scores
- **Threshold Comparison**: Results against configurable thresholds
- **Method Comparison**: Side-by-side analysis of different methods

## How It Works

1. **Face Detection**: Extract face regions from input images
2. **Feature Extraction**: Generate high-dimensional embeddings using deep learning
3. **Distance Calculation**: Compute similarity between embeddings using various metrics
4. **Verification**: Compare distance against threshold to determine match
5. **Visualization**: Display results with interactive charts and analysis

## 🛠️ Development

### Code Style

The project follows PEP8 standards with:
- Type hints for all functions
- Comprehensive docstrings
- Black code formatting
- Flake8 linting

### Adding New Methods

To add a new face verification method:

1. Implement the method in `FaceVerificationSystem`
2. Add configuration options in `config.yaml`
3. Update the web interface in `web_app/app.py`
4. Add tests in `tests/`

## Future Enhancements

- [ ] Real-time face detection from webcam
- [ ] Face recognition (identification) capabilities
- [ ] Advanced visualization features
- [ ] Model fine-tuning capabilities
- [ ] API endpoints for integration
- [ ] Mobile app interface
- [ ] Cloud deployment support

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) - Face recognition library
- [DeepFace](https://github.com/serengil/deepface) - Deep face analysis library
- [Hugging Face](https://huggingface.co/) - Transformers library
- [Streamlit](https://streamlit.io/) - Web application framework

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation


**Note**: This system is designed for educational and research purposes. For production use, consider additional security measures, data privacy compliance, and performance optimization.
# Face-Verification-System
