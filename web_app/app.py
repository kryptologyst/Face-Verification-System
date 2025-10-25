"""
Streamlit Web Interface for Face Verification System

This module provides a user-friendly web interface for the face verification system
using Streamlit, allowing users to upload images and perform face verification.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from face_verification import FaceVerificationSystem
from data_generator import SyntheticFaceGenerator, MockDatabase, create_sample_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Face Verification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'verification_system' not in st.session_state:
    st.session_state.verification_system = None
if 'database' not in st.session_state:
    st.session_state.database = None
if 'sample_data_created' not in st.session_state:
    st.session_state.sample_data_created = False


def initialize_system():
    """Initialize the face verification system."""
    if st.session_state.verification_system is None:
        try:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            st.session_state.verification_system = FaceVerificationSystem(str(config_path))
            st.session_state.database = MockDatabase("data/mock_database.json")
            logger.info("Face verification system initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
            logger.error(f"System initialization error: {e}")


def create_sample_data():
    """Create sample dataset if it doesn't exist."""
    if not st.session_state.sample_data_created:
        try:
            with st.spinner("Creating sample dataset..."):
                create_sample_dataset("data", num_persons=10)
                st.session_state.sample_data_created = True
                st.success("Sample dataset created successfully!")
        except Exception as e:
            st.error(f"Failed to create sample dataset: {e}")
            logger.error(f"Sample data creation error: {e}")


def display_image_comparison(image1: np.ndarray, image2: np.ndarray, 
                           results: Dict[str, Dict]) -> None:
    """Display image comparison with results."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image1, caption="Image 1", use_column_width=True)
    
    with col2:
        st.image(image2, caption="Image 2", use_column_width=True)
    
    # Display results
    st.subheader("🔍 Verification Results")
    
    # Create metrics for each method
    for method, result in results.items():
        if result['success']:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label=f"{method.upper()} Distance",
                    value=f"{result['distance']:.4f}",
                    delta=f"Threshold: {result['threshold']:.4f}"
                )
            
            with col2:
                st.metric(
                    label=f"{method.upper()} Confidence",
                    value=f"{result['confidence']:.2%}"
                )
            
            with col3:
                status = "✅ Match" if result['is_match'] else "❌ No Match"
                st.metric(
                    label=f"{method.upper()} Result",
                    value=status
                )
            
            with col4:
                # Color-coded result
                if result['is_match']:
                    st.markdown('<div class="metric-card success-message">SAME PERSON</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-card error-message">DIFFERENT PEOPLE</div>', 
                              unsafe_allow_html=True)
        else:
            st.error(f"{method.upper()}: {result['error']}")


def create_results_visualization(results: Dict[str, Dict]) -> None:
    """Create interactive visualization of results."""
    st.subheader("📊 Results Visualization")
    
    # Prepare data for plotting
    methods = []
    distances = []
    thresholds = []
    confidences = []
    matches = []
    
    for method, result in results.items():
        if result['success']:
            methods.append(method.upper())
            distances.append(result['distance'])
            thresholds.append(result['threshold'])
            confidences.append(result['confidence'])
            matches.append(result['is_match'])
    
    if not methods:
        st.warning("No successful verification results to visualize")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distance vs Threshold', 'Confidence Scores', 
                       'Match Results', 'Method Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Distance vs Threshold
    fig.add_trace(
        go.Bar(name='Distance', x=methods, y=distances, marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Threshold', x=methods, y=thresholds, marker_color='lightcoral'),
        row=1, col=1
    )
    
    # Confidence scores
    fig.add_trace(
        go.Bar(name='Confidence', x=methods, y=confidences, marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Match results
    match_values = [1 if match else 0 for match in matches]
    fig.add_trace(
        go.Bar(name='Match (1=Yes, 0=No)', x=methods, y=match_values, 
               marker_color=['green' if m else 'red' for m in matches]),
        row=2, col=1
    )
    
    # Method comparison radar chart
    fig.add_trace(
        go.Scatterpolar(
            r=confidences,
            theta=methods,
            fill='toself',
            name='Confidence',
            marker_color='rgba(0,100,80,0.3)'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Face Verification Analysis Dashboard"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">🔍 Face Verification System</h1>', 
                unsafe_allow_html=True)
    
    # Initialize system
    initialize_system()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Method selection
    available_methods = []
    if st.session_state.verification_system:
        if st.session_state.verification_system.models.get('face_recognition'):
            available_methods.append('face_recognition')
        if st.session_state.verification_system.models.get('deepface'):
            available_methods.append('deepface')
        if st.session_state.verification_system.models.get('transformers'):
            available_methods.append('transformers')
    
    selected_methods = st.sidebar.multiselect(
        "Select verification methods:",
        available_methods,
        default=available_methods
    )
    
    # Distance method selection
    distance_method = st.sidebar.selectbox(
        "Distance calculation method:",
        ["euclidean", "cosine", "manhattan"],
        index=0
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Upload Images", "🎲 Sample Data", "📊 Database", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Images for Verification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image 1")
            uploaded_file1 = st.file_uploader(
                "Choose first image",
                type=['jpg', 'jpeg', 'png'],
                key="image1"
            )
            
            if uploaded_file1 is not None:
                image1 = Image.open(uploaded_file1)
                image1_array = np.array(image1)
                st.image(image1, caption="Uploaded Image 1", use_column_width=True)
        
        with col2:
            st.subheader("Image 2")
            uploaded_file2 = st.file_uploader(
                "Choose second image",
                type=['jpg', 'jpeg', 'png'],
                key="image2"
            )
            
            if uploaded_file2 is not None:
                image2 = Image.open(uploaded_file2)
                image2_array = np.array(image2)
                st.image(image2, caption="Uploaded Image 2", use_column_width=True)
        
        # Verify button
        if st.button("🔍 Verify Faces", disabled=not (uploaded_file1 and uploaded_file2)):
            if uploaded_file1 and uploaded_file2 and st.session_state.verification_system:
                try:
                    # Save uploaded files temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1:
                        image1.save(tmp1.name)
                        tmp1_path = tmp1.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
                        image2.save(tmp2.name)
                        tmp2_path = tmp2.name
                    
                    # Perform verification
                    with st.spinner("Performing face verification..."):
                        results = st.session_state.verification_system.verify_faces(
                            tmp1_path, tmp2_path, selected_methods
                        )
                    
                    # Display results
                    display_image_comparison(image1_array, image2_array, results)
                    create_results_visualization(results)
                    
                    # Clean up temporary files
                    os.unlink(tmp1_path)
                    os.unlink(tmp2_path)
                    
                except Exception as e:
                    st.error(f"Verification failed: {e}")
                    logger.error(f"Verification error: {e}")
            else:
                st.warning("Please upload both images to perform verification")
    
    with tab2:
        st.header("Sample Data Demo")
        
        if st.button("🎲 Generate Sample Dataset"):
            create_sample_data()
        
        if st.session_state.sample_data_created:
            st.success("Sample dataset is ready!")
            
            # Show sample persons
            if st.session_state.database:
                person_ids = list(st.session_state.database.data['persons'].keys())
                
                if person_ids:
                    st.subheader("Available Sample Persons")
                    
                    # Person selection
                    selected_person1 = st.selectbox("Select Person 1:", person_ids)
                    selected_person2 = st.selectbox("Select Person 2:", person_ids)
                    
                    if st.button("🔍 Verify Sample Faces"):
                        try:
                            person1_data = st.session_state.database.get_person(selected_person1)
                            person2_data = st.session_state.database.get_person(selected_person2)
                            
                            if person1_data and person2_data and person1_data['image_paths'] and person2_data['image_paths']:
                                img1_path = person1_data['image_paths'][0]
                                img2_path = person2_data['image_paths'][0]
                                
                                # Load images for display
                                img1 = cv2.imread(img1_path)
                                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                                img2 = cv2.imread(img2_path)
                                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                                
                                # Perform verification
                                with st.spinner("Performing verification..."):
                                    results = st.session_state.verification_system.verify_faces(
                                        img1_path, img2_path, selected_methods
                                    )
                                
                                # Display results
                                display_image_comparison(img1, img2, results)
                                create_results_visualization(results)
                                
                                # Add to database
                                is_match = any(result.get('is_match', False) for result in results.values() if result.get('success'))
                                st.session_state.database.add_verification_record(
                                    selected_person1, selected_person2, results, is_match
                                )
                                
                            else:
                                st.error("Selected persons don't have images")
                                
                        except Exception as e:
                            st.error(f"Verification failed: {e}")
                            logger.error(f"Sample verification error: {e}")
    
    with tab3:
        st.header("Database Management")
        
        if st.session_state.database:
            # Show database statistics
            persons_count = len(st.session_state.database.data['persons'])
            verifications_count = len(st.session_state.database.data['verification_history'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Persons", persons_count)
            with col2:
                st.metric("Total Verifications", verifications_count)
            
            # Show recent verifications
            if verifications_count > 0:
                st.subheader("Recent Verifications")
                recent_verifications = st.session_state.database.get_verification_history(limit=10)
                
                for verification in recent_verifications:
                    with st.expander(f"Verification: {verification['person1_id']} vs {verification['person2_id']}"):
                        st.write(f"**Result:** {'✅ Match' if verification['is_match'] else '❌ No Match'}")
                        st.write(f"**Methods used:** {list(verification['results'].keys())}")
                        
                        # Show detailed results
                        for method, result in verification['results'].items():
                            if result['success']:
                                st.write(f"- **{method}:** Distance={result['distance']:.4f}, "
                                        f"Confidence={result['confidence']:.2%}")
    
    with tab4:
        st.header("About the Face Verification System")
        
        st.markdown("""
        ## 🔍 Face Verification System
        
        This is a modern, state-of-the-art face verification system that can determine 
        whether two face images belong to the same person or different people.
        
        ### Features:
        - **Multiple Backends**: Supports face_recognition, DeepFace, and Hugging Face transformers
        - **Configurable Thresholds**: Adjustable verification thresholds for each method
        - **Multiple Distance Metrics**: Euclidean, cosine, and Manhattan distance calculations
        - **Interactive Web Interface**: Easy-to-use Streamlit interface
        - **Sample Data Generation**: Synthetic face generation for testing
        - **Database Integration**: Mock database for storing verification history
        - **Visualization**: Interactive charts and result analysis
        
        ### How it Works:
        1. **Face Detection**: Extract face regions from input images
        2. **Feature Extraction**: Generate high-dimensional embeddings using deep learning
        3. **Distance Calculation**: Compute similarity between embeddings
        4. **Verification**: Compare distance against threshold to determine match
        
        ### Supported Methods:
        - **face_recognition**: Traditional face recognition library
        - **DeepFace**: Advanced face analysis with multiple models
        - **Transformers**: Modern vision transformer models
        
        ### Usage:
        1. Upload two face images or use sample data
        2. Select verification methods
        3. Click "Verify Faces" to get results
        4. View detailed analysis and visualizations
        
        ### Technical Details:
        - Built with Python 3.8+
        - Uses modern libraries: transformers, DeepFace, face_recognition
        - Follows PEP8 coding standards
        - Comprehensive type hints and documentation
        - Configurable via YAML files
        - Logging and error handling
        """)


if __name__ == "__main__":
    main()
