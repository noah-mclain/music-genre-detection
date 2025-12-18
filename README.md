# Music Genre Detection

A deep learning project for automatic music genre classification using a CNN-LSTM-Attention neural network architecture. This system can classify music into 10 genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock.

## Overview

This project implements an advanced deep learning model that combines:
- **Convolutional Neural Networks (CNN)** for feature extraction from spectrograms
- **Long Short-Term Memory (LSTM)** networks for temporal pattern recognition
- **Multi-head Attention mechanisms** for focusing on important temporal segments

The model processes mel-spectrograms of audio files to predict the genre with high accuracy. It includes both a command-line interface and a web-based UI for easy interaction.

## Quick Start

Get started in just a few commands:

```bash
# Clone the repository
git clone https://github.com/noah-mclain/music-genre-detection.git
cd music-genre-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the web interface
python app.py
```

Then open http://localhost:5001 in your browser!

## Features

- 🎵 **Automatic Genre Classification**: Classifies music into 10 genres with attention-based deep learning
- 🎯 **Explainable AI**: Includes SHAP, LIME, Grad-CAM, and Integrated Gradients for model interpretability
- 🌐 **Web Interface**: Flask-based web UI for easy audio upload and prediction
- 📊 **Audio Processing**: Mel-spectrogram extraction with normalization and augmentation
- 📈 **Comprehensive Testing**: Unit tests for all major components
- 🔍 **Code Quality**: Includes linting, type checking, and formatting tools
- 📚 **Documentation**: Sphinx-based API documentation

## Supported Audio Formats

- MP3
- WAV
- M4A
- FLAC

Maximum file size: 50MB

## Project Structure

```
music-genre-detection/
├── src/
│   ├── models.py              # CNN-LSTM-Attention model architecture
│   ├── inference_utils.py     # Audio processing and model inference
│   ├── download_data.py       # Data download utilities
│   ├── database.py            # Database operations
│   ├── logging_config.py      # Logging configuration
│   ├── mock_classifier.py     # Mock classifier for testing
│   ├── explainability.py      # XAI model explanations
│   ├── GradCam.py             # Grad-CAM visualization
│   ├── IG.py                  # Integrated Gradients
│   └── preprocessed_cache/    # Cached preprocessed spectrograms
├── tests/
│   ├── test_models.py         # Model tests
│   ├── test_inference_utils.py
│   ├── test_database.py
│   └── conftest.py
├── templates/                 # HTML templates for web UI
│   └── index.html
├── static/                    # CSS and JavaScript assets
│   ├── css/
│   └── js/
├── docs/                      # Sphinx documentation
│   ├── conf.py
│   ├── index.rst
│   └── _build/
├── app.py                     # Flask application
├── web_ui.py                  # Alternative web UI
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone noah-mclain/music-genre-detection
   cd music-genre-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

Start the Flask web application:

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5001` (or the configured port).

Upload an audio file and the model will predict its genre with confidence scores.

## Model Architecture

The CNN-LSTM-Attention model consists of:

1. **CNN Feature Extractor**
   - 4 convolutional blocks with batch normalization and ReLU activation
   - Max pooling and dropout for regularization
   - Processes mel-spectrograms (128 mel-bins × 130 time steps)

2. **LSTM Layers**
   - 2 bidirectional LSTM layers with configurable hidden dimension
   - Captures temporal dependencies in the spectrogram from both directions

3. **Attention Mechanism**
   - Multi-head attention over LSTM outputs
   - Learns to focus on important time segments
   - Global average pooling for final representation

4. **Classification Head**
   - Fully connected layers
   - Softmax output for 10 genres

## Audio Processing Pipeline

1. **Load Audio**: Load audio file at 22050 Hz sample rate
2. **Extract Mel-Spectrogram**: 
   - FFT size: 2048
   - Hop length: 512
   - Mel bins: 128
3. **Normalize**: Convert to dB scale with z-score normalization
4. **Pad/Trim**: Fixed length of 130 time steps

## Configuration

Key configuration parameters in `app.py`:

```python
SAMPLE_RATE = 22050        # Audio sample rate
N_MELS = 128              # Number of mel frequency bins
DURATION = 30             # Audio duration in seconds
N_FFT = 2048              # FFT window size
HOP_LENGTH = 512          # FFT hop length
SEGMENT_LENGTH = 130      # Fixed segment length
MAX_FILE_SIZE = 50MB      # Maximum upload file size
```

Environment variables:
- `FLASK_DEBUG`: Enable/disable Flask debug mode
- `MODEL_PATH`: Path to trained model weights
- `USE_MOCK`: Use mock classifier for testing

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

With coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Code Quality

Run code quality checks:

```bash
./run_quality_checks.sh
```

This includes:
- **black**: Code formatting
- **isort**: Import sorting
- **pylint**: Linting
- **mypy**: Type checking

## Model Explainability

The project includes two explainability techniques:

- **Grad-CAM**: Gradient-based visual attention maps
- **Integrated Gradients**: Feature attribution via gradients

See `src/explainability.py` for implementation details.

## Documentation

Build Sphinx documentation:

```bash
cd docs
make html
```

Or if you don't have make:

```bash
cd docs
sphinx-build -b html . _build/html
```

View documentation at `docs/_build/html/index.html`

## Dependencies

### Core Libraries
- PyTorch 2.1.0+
- LibROSA 0.10.0+
- NumPy 1.24.0+
- SciPy 1.10.0+

### Web Framework
- Flask 2.3.0+
- Flask-CORS 3.0.0+

### ML/Data Processing
- Scikit-learn 1.3.0+
- Matplotlib 3.8.0+
- Seaborn 0.12.2+

### Explainability
- Captum 0.7.0+
- OpenCV 4.8.0+

### Code Quality
- Black, Pylint, MyPy, isort
- Pytest for testing

For complete dependencies, see [requirements.txt](requirements.txt).

## Authors

- **Nada Ayman** (nadamo.cs@gmai.co)
- **Omar Khaled** (omarkhaledali64@gmail.com)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure that:

1. Code follows the project's style guide (Black, isort)
2. All tests pass: `pytest tests/`
3. Code quality checks pass: `./run_quality_checks.sh`
4. Type annotations are included
5. Documentation is updated

## Acknowledgments

This project uses the GTZAN Music Genre Dataset and implements state-of-the-art deep learning techniques for music genre classification.