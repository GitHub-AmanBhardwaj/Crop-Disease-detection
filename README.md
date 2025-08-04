# Crop Disease Detection: MobileNetV2 and EfficientNetB3 with Generative AI Integration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras 2.x](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![Flask 2.0+](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![NumPy 1.21+](https://img.shields.io/badge/numpy-1.21+-blue.svg)](https://numpy.org/)


This repository contains the **training notebooks**, **pre-trained models**, and **Flask web application** for crop disease detection research. The work demonstrates two different CNN architectures: **MobileNetV2** (designed for edge devices) and **EfficientNetB3** (designed for cloud processing), both integrated with **Google's Gemini API** for AI-generated treatment recommendations.

**Note**: This repository contains the individual model components and a demonstration web application. The full hybrid edge-cloud implementation described in our research paper would require additional deployment infrastructure for production use and the prototype code will be uploaded soon.

### Key Features

- **Two Trained Models**: MobileNetV2 (93.54% accuracy) and EfficientNetB3 (96.56% accuracy)
- **Complete Training Pipeline**: Jupyter notebooks with full training code and analysis
- **27 Disease Classes**: Covers 6 major crops (Apple, Corn, Grape, Potato, Strawberry, Tomato)
- **AI-Powered Recommendations**: Flask app with Google Gemini API integration
- **Ready-to-Use**: Pre-trained models and working web application included

## Dataset

**PlantVillage Dataset Subset**
- **Total Images**: 32,962 colored leaf images
- **Classes**: 27 disease and healthy categories
- **Crops**: Apple, Corn, Grape, Potato, Strawberry, Tomato
- **Split**: 64% Training, 16% Validation, 20% Testing
- **Resolution**: 224×224 (MobileNetV2), 300×300 (EfficientNetB3)

## Model Architectures

### MobileNetV2 (Edge-Optimized)
- **Input**: 224×224×3 RGB images
- **Base**: Pre-trained ImageNet weights (frozen)
- **Custom Layers**: Global Average Pooling → Dense(512) → Dropout(0.5) → Dense(27)
- **Parameters**: 2.93M (669K trainable)
- **Training**: 50 epochs, batch size 32

### EfficientNetB3 (Cloud-Optimized)
- **Input**: 300×300×3 RGB images
- **Base**: Pre-trained ImageNet weights with fine-tuning (last 30 layers)
- **Custom Layers**: Global Average Pooling → Dense(1024) → Dense(512) → Dense(27)
- **Parameters**: 12.90M (2.12M+ trainable)
- **Training**: 25 epochs + 10 fine-tuning epochs, batch size 32

## Results

| Model | Test Accuracy | Validation Accuracy | Test Loss | Weighted F1-Score |
|-------|---------------|---------------------|-----------|-------------------|
| MobileNetV2 | 93.54% | 93.42% | 0.1980 | 0.9356 |
| EfficientNetB3 | 96.56% | 96.25% | 0.1029 | 0.9659 |

## Quick Start

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Flask
- Google Gemini API Key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/GitHub-AmanBhardwaj/Crop-Disease-detection.git
cd Crop-Disease-detection
```

2. **Install dependencies**
```bash
pip install tensorflow flask numpy matplotlib seaborn scikit-learn pillow pandas google-generativeai python-dotenv
```

3. **Set up environment variables**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

## Usage

### Running the Flask Web Application

```bash
cd flask_app
python app.py
```

Access the application at `http://localhost:5000`

### Using the Training Notebooks

1. **Reproduce Training**: Run the notebooks to train models from scratch
2. **Analysis**: Explore model performance and visualizations

```bash
jupyter notebook notebooks/MobileNetV2_Training.ipynb
jupyter notebook notebooks/EfficientNetB3_Training.ipynb
```

## Repository Contents

```
Crop-Disease-detection/
├── models/
│   ├── best_disease_model.keras          # Trained MobileNetV2 model
│   └── best_disease_model_b3.keras       # Trained EfficientNetB3 model
├── notebooks/
│   ├── mobilenetv2.ipynb        # MobileNetV2 training & evaluation
│   └── efficientnetb3.ipynb     # EfficientNetB3 training & evaluation
├── flask_app/
│   ├── app.py                            # Flask web application
│   ├── templates/                        # HTML templates
│   └── static/                           # CSS, JS, and images
├── LICENSE                               # MIT License
└── README.md                            # This file
```

## Flask Web Application

The included Flask application demonstrates real-time crop disease detection:

- **Disease Classification**: Upload leaf images for instant disease prediction
- **Model Selection**: Uses pre-trained models for classification
- **Confidence Scoring**: Displays prediction confidence levels
- **AI Recommendations**: Integrates with Google Gemini API for treatment advice
- **User-Friendly Interface**: Simple web interface for easy interaction

## Research Context

This repository supports our research on **"Hybrid Edge–Cloud Crop Disease Diagnosis"**. While the full hybrid implementation would require additional deployment infrastructure, this repository provides:

- **Model Components**: Both edge-optimized (MobileNetV2) and cloud-optimized (EfficientNetB3) models
- **Training Pipeline**: Complete notebooks showing the training process
- **Demonstration Application**: Flask app showing practical implementation
- **Performance Analysis**: Detailed evaluation of both approaches

## Use Cases

- **Research Reproduction**: Use notebooks to reproduce our training results
- **Model Comparison**: Compare edge vs cloud model architectures
- **Educational Purpose**: Learn about crop disease classification using deep learning
- **Application Development**: Use pre-trained models in your own applications
- **Web Interface Demo**: See practical implementation with Gemini AI integration

## 🔧 Configuration

### Required Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Gemini API Setup
1. Get API key from [Google AI Studio](https://developers.google.com/gemini)
2. Set the environment variable
3. The Flask app will generate treatment recommendations automatically

## Authors

- **Aman Bhardwaj** - *Lead Developer* - [GitHub](https://github.com/GitHub-AmanBhardwaj)
  - Email: whyamanbhardwaj@gmail.com
- **Jeet Bhardwaj** - *Co-Developer* - Email: thejeetbhardwaj@gmail.com
- **Sumit Dhariwal** - *Supervisor* - Email: sumitdhariwal22@gmail.com

## Affiliation

**Centre for AI, Madhav Institute of Technology and Science (MITS-DU), Gwalior, India**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:
- Email: whyamanbhardwaj@gmail.com
- Issues: [GitHub Issues](https://github.com/GitHub-AmanBhardwaj/Crop-Disease-detection/issues)

⭐ **Star this repository if you find it helpful!** 
