# Hybrid Edge-Cloud Crop Disease Diagnosis: MobileNetV2 and EfficientNetB3 for On-Device Detection with Generative AI Integration


This repository contains the implementation of a **Hybrid Edge-Cloud Crop Disease Diagnosis System** that combines the efficiency of on-device inference with the accuracy of cloud-based processing. The system uses **MobileNetV2** for lightweight edge detection and **EfficientNetB3** for high-accuracy cloud processing, integrated with **Google's Gemini API** for personalized treatment recommendations.

### 🎯 Key Features

- **Hybrid Architecture**: Combines edge computing (MobileNetV2) with cloud processing (EfficientNetB3)
- **High Accuracy**: EfficientNetB3 achieves 96.56% test accuracy, MobileNetV2 achieves 93.54%
- **27 Disease Classes**: Covers 6 major crops (Apple, Corn, Grape, Potato, Strawberry, Tomato)
- **AI-Powered Recommendations**: Integration with Google Gemini API for treatment advice
- **Web Interface**: Real-time Flask-based web application
- **Production Ready**: Complete deployment pipeline with model optimization

## 🌱 Dataset

**PlantVillage Dataset Subset**
- **Total Images**: 32,962 colored leaf images
- **Classes**: 27 disease and healthy categories
- **Crops**: Apple, Corn, Grape, Potato, Strawberry, Tomato
- **Split**: 64% Training, 16% Validation, 20% Testing
- **Resolution**: 224×224 (MobileNetV2), 300×300 (EfficientNetB3)

## 🏗️ Model Architectures

### MobileNetV2 (Edge Model)
- **Input**: 224×224×3 RGB images
- **Base**: Pre-trained ImageNet weights (frozen)
- **Custom Layers**: Global Average Pooling → Dense(512) → Dropout(0.5) → Dense(27)
- **Parameters**: 2.93M (669K trainable)
- **Training**: 50 epochs, batch size 32

### EfficientNetB3 (Cloud Model)
- **Input**: 300×300×3 RGB images
- **Base**: Pre-trained ImageNet weights with fine-tuning (last 30 layers)
- **Custom Layers**: Global Average Pooling → Dense(1024) → Dense(512) → Dense(27)
- **Parameters**: 12.90M (2.12M+ trainable)
- **Training**: 25 epochs + 10 fine-tuning epochs, batch size 32

## 📊 Results

| Model | Test Accuracy | Validation Accuracy | Test Loss | Weighted F1-Score |
|-------|---------------|---------------------|-----------|-------------------|
| MobileNetV2 | 93.54% | 93.42% | 0.1980 | 0.9356 |
| EfficientNetB3 | 96.56% | 96.25% | 0.1029 | 0.9659 |

## 🚀 Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Flask
- Google Gemini API Key

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/GitHub-AmanBhardwaj/Crop-Disease-detection.git
cd Crop-Disease-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

4. **Download pre-trained models** (Available in `models/` directory)
   - `best_disease_model.keras` (MobileNetV2)
   - `best_disease_model_b3.keras` (EfficientNetB3)

## 💻 Usage

### Training Models

1. **Train MobileNetV2**
```bash
python train_mobilenet.py
```

2. **Train EfficientNetB3**
```bash
python train_efficientnet.py
```

### Running the Web Application

```bash
python app.py
```

Access the application at `http://localhost:5000`

### Using the Hybrid System

```python
from src.hybrid_inference import HybridClassifier

# Initialize the hybrid classifier
classifier = HybridClassifier(
    mobilenet_path="models/best_disease_model.keras",
    efficientnet_path="models/best_disease_model_b3.keras",
    confidence_threshold=0.8
)

# Classify an image
result = classifier.predict("path/to/leaf_image.jpg")
print(f"Disease: {result['disease']}")
print(f"Confidence: {result['confidence']}")
print(f"Recommendations: {result['recommendations']}")
```

## 📁 Repository Structure

```
Crop-Disease-detection/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
│   ├── best_disease_model.keras
│   ├── best_disease_model_b3.keras
│   └── model_configs/
├── notebooks/
│   ├── MobileNetV2_Training.ipynb
│   ├── EfficientNetB3_Training.ipynb
│   └── Data_Analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_mobilenet.py
│   ├── train_efficientnet.py
│   ├── hybrid_inference.py
│   └── utils.py
├── web/
│   ├── app.py
│   ├── templates/
│   └── static/
├── results/
│   ├── plots/
│   ├── metrics/
│   └── confusion_matrices/
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔧 Configuration

### Model Parameters

**MobileNetV2**
- Learning Rate: 0.0001
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Data Augmentation: Rotation, Shift, Shear, Zoom, Flip

**EfficientNetB3**
- Learning Rate: 0.0001 (initial), 0.00001 (fine-tuning)
- Optimizer: Adam
- Loss: Weighted Categorical Crossentropy
- Class Weights: Computed for imbalanced classes

## 🤖 API Integration

### Google Gemini API Setup

1. Get API key from [Google AI Studio](https://developers.google.com/gemini)
2. Set environment variable: `GEMINI_API_KEY`
3. The system automatically generates treatment recommendations based on disease classification

## 📈 Performance Metrics

### Class-wise Performance (Top/Bottom performers)

**Best Performing Classes:**
- Corn Healthy: 100% F1-score (both models)
- Grape Healthy: 100% F1-score (both models)
- Apple Black Rot: 99-100% F1-score

**Challenging Classes:**
- Tomato Early Blight: 79% (MobileNetV2), 90% (EfficientNetB3)
- Tomato Target Spot: 81% (MobileNetV2), 91% (EfficientNetB3)
- Corn Gray Leaf Spot: 80% (MobileNetV2), 90% (EfficientNetB3)

## 🎯 Use Cases

- **Mobile Agriculture Apps**: Real-time disease detection on smartphones
- **IoT Farm Monitoring**: Edge devices with cloud backup processing
- **Agricultural Extension Services**: Professional diagnostic tools
- **Research Applications**: Disease pattern analysis and monitoring

## 🚀 Deployment

### Docker Deployment
```bash
docker build -t crop-disease-detector .
docker run -p 5000:5000 -e GEMINI_API_KEY=your_key crop-disease-detector
```

### Cloud Deployment
- Supports deployment on AWS, Google Cloud, Azure
- Edge model optimized for mobile/IoT devices
- Cloud model suitable for high-throughput processing

## 📚 Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{bhardwaj2025hybrid,
  title={Hybrid Edge–Cloud Crop Disease Diagnosis: MobileNetV2 and EfficientNetB3 for On-Device Detection with Generative AI Integration},
  author={Bhardwaj, Aman and Bhardwaj, Jeet and Dhariwal, Sumit},
  journal={[Journal Name]},
  year={2025},
  publisher={MDPI}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Aman Bhardwaj** - *Lead Developer* - [GitHub](https://github.com/GitHub-AmanBhardwaj)
  - Email: whyamanbhardwaj@gmail.com
- **Jeet Bhardwaj** - *Co-Developer* - Email: thejeetbhardwaj@gmail.com
- **Sumit Dhariwal** - *Supervisor* - Email: sumitdhariwal22@gmail.com

## 🏛️ Affiliation

**Centre for AI, Madhav Institute of Technology and Science (MITS-DU), Gwalior, India**

## 🙏 Acknowledgments

- PlantVillage dataset creators
- Google for Gemini API access
- TensorFlow and Keras teams
- Open source community

## 📞 Support

For questions and support:
- 📧 Email: whyamanbhardwaj@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/GitHub-AmanBhardwaj/Crop-Disease-detection/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/GitHub-AmanBhardwaj/Crop-Disease-detection/discussions)

⭐ **Star this repository if you find it helpful!** ⭐
