# CropPredict: AI-Powered Crop Disease Detection

CropPredict is an academic project that leverages deep learning to detect diseases in crops using leaf images. The project focuses on six crops—Apple, Corn, Grapes, Potato, Strawberry, and Tomato—and employs convolutional neural networks (CNNs) like MobileNetV2 and EfficientNetB3. A Flask-based web application provides an interface for uploading images and viewing predictions, while a chatbot offers insights on disease prevention and management. This repository contains the model code, Flask app, and HTML templates for the CropPredict platform.

## Table of Contents

- Project Overview
- Features
- Repository Structure
- Prerequisites
- Installation
- Model Training
- Running the Flask App
- Usage
- Templates
- Contributing
- License
- Acknowledgements

## Project Overview

CropPredict is developed to explore the application of AI in agriculture, specifically for detecting crop diseases through image analysis. The project uses a dataset of leaf images to train CNN models, achieving high accuracy in classifying diseases such as Tomato___Late_blight and Apple___healthy. The web app, built with Flask, allows users to upload leaf images for prediction and interact with a chatbot for disease management advice. The project aims to support agricultural research and education by demonstrating AI’s potential in crop health monitoring.

## Features

- **Disease Detection**: Classifies diseases in six crops using MobileNetV2 and EfficientNetB3 models.
- **Web Interface**: Upload leaf images via a drag-and-drop interface and view prediction results.
- **Chatbot**: Provides AI-driven suggestions for disease prevention and management.
- **Responsive Design**: Templates (`predict.html`, `about.html`) are mobile-friendly with consistent navigation.
- **Academic Focus**: Emphasizes research-oriented goals, including dataset expansion and model improvement.

## Repository Structure

```
CropPredict/
├── models/
│   ├── train_model.py              # Script to train MobileNetV2 and EfficientNetB3 models
│   ├── best_disease_model.keras    # Pre-trained MobileNetV2 model
│   ├── best_disease_model_b3.keras # Pre-trained EfficientNetB3 model
├── app/
│   ├── app.py                     # Flask application
│   ├── static/
│   │   ├── img/                   # Images (e.g., crop-field.jpg, ai-model.jpg)
│   │   ├── uploads/               # Directory for uploaded images
│   ├── templates/
│   │   ├── predict1.html           # Prediction page template
│   │   ├── predict2.html           # Prediction page template
│   │   ├── about.html             # About page template
│   │   ├── bot.html               # Chatbot page template
├── data/
│   ├── dataset/                   # Dataset of leaf images (not included, see notes)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # License file
```

**Note**: The `dataset/` directory is not included due to size constraints. Download the dataset from PlantVillage or a similar source and place it in `data/dataset/`.

## Prerequisites

- **Python**: 3.8 or higher
- **Dependencies**: Listed in `requirements.txt`
- **Dataset**: Leaf image dataset for training (e.g., PlantVillage)
- **Hardware**: GPU recommended for model training
- **Browser**: Chrome, Firefox, or Safari for web app

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/<your-username>/CropPredict.git
   cd CropPredict
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:

   ```
   flask==2.3.2
   tensorflow==2.12.0
   numpy==1.24.3
   pillow==9.5.0
   werkzeug==2.3.6
   ```

4. **Download the Dataset**:

   - Obtain the PlantVillage dataset or equivalent.
   - Extract and place it in `data/dataset/` with subfolders for each crop and disease (e.g., `Tomato___Late_blight`, `Apple___healthy`).

5. **Verify Model Files**:

   - Ensure `best_disease_model.keras` and `best_disease_model_b3.keras` are in `models/`.
   - Alternatively, train models using `train_model.py` (see Model Training).

## Model Training

The project uses MobileNetV2 and EfficientNetB3 for disease classification. To train the models:

1. **Prepare the Dataset**:

   - Organize images in `data/dataset/` with subfolders for each class (e.g., `Tomato___Late_blight`).
   - Example structure:

     ```
     data/dataset/
     ├── Apple___healthy/
     ├── Apple___Black_rot/
     ├── Tomato___Late_blight/
     ...
     ```

2. **Run the Training Script**:

   ```bash
   python models/train_model.py
   ```

   - Update `train_model.py` with paths to your dataset and model output directory.
   - Parameters (example):
     - Batch size: 32
     - Epochs: 50
     - Image size: 224x224
     - Optimizer: Adam
     - Loss: Categorical Crossentropy

3. **Output**:

   - Trained models saved as `best_disease_model.keras` (MobileNetV2) and `best_disease_model_b3.keras` (EfficientNetB3) in `models/`.

**Note**: Training requires significant computational resources. Use a GPU or cloud service (e.g., Google Colab) for faster results. Pre-trained models are provided for convenience.

## Running the Flask App

1. **Set Environment Variables** (if applicable):

   - For chatbot integration, set API keys (e.g., for Gemini API) in `app.py` or a `.env` file.

   ```bash
   export GEMINI_API_KEY='your-api-key'  # On Windows: set GEMINI_API_KEY=your-api-key
   ```

2. **Start the Flask App**:

   ```bash
   python app/app.py
   ```

   - The app runs on `http://localhost:8000` by default.

3. **Directory Setup**:

   - Ensure `static/uploads/` exists and is writable for image uploads.
   - Verify `static/img/` contains images referenced in templates (e.g., `crop-field.jpg`, `ai-model.jpg`).

## Usage

1. **Access the Web App**:

   - Open `http://localhost:8000` in a browser.
   - Navigate to:
     - **Home/Predict**: `/predict1` (MobileNetV2) or `/predict2` (EfficientNetB3) for image uploads.
     - **Chatbot**: `/bot` for disease management advice.
     - **About**: `/about` for project details.

2. **Upload an Image**:

   - On `/predict1` or `/predict2`, drag-and-drop or browse a leaf image (PNG, JPG, &lt;16MB).
   - View prediction results (e.g., “Tomato___Late_blight (Confidence: 0.87)”).
   - For non-healthy predictions, click “Prevention and Cure” for chatbot suggestions.

3. **Interact with the Chatbot**:

   - On `/bot`, ask questions about crop diseases or prevention (e.g., “How to treat Tomato Late Blight?”).
   - Responses are generated using an AI model (e.g., Gemini).

4. **Explore the About Page**:

   - Learn about the project’s academic goals, technical approach (CNNs, dataset), and future plans.

## Templates

The `templates/` directory contains HTML files for the web app:

- **predict.html**:
  - Main page for uploading leaf images and viewing predictions.
  - Features drag-and-drop, image preview, and prevention button.
  - Endpoints: `/predict1` (MobileNetV2), `/predict2` (EfficientNetB3).
- **about.html**:
  - Describes the project’s purpose, technical approach, and future goals.
  - Sections: Project Overview, Technical Approach, Future Goals.
- **bot.html**:
  - Interface for the chatbot, allowing users to query disease prevention and management.

**Styling**:

- Uses Font Awesome for icons and a custom CSS palette (`--primary-color: #4361ee`, `--dark-color: #212529`).
- Responsive design with mobile-friendly navigation and layouts.

## Contributing

Contributions are welcome! To contribute:

1. **Fork the Repository**:

   ```bash
   git clone https://github.com/<your-username>/CropPredict.git
   ```

2. **Create a Branch**:

   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make Changes**:

   - Add new features (e.g., model improvements, UI enhancements).
   - Fix bugs (e.g., upload issues in `predict.html`).
   - Update documentation.

4. **Submit a Pull Request**:

   - Push changes to your fork and create a PR with a clear description.

**Guidelines**:

- Follow Python PEP 8 style guidelines.
- Test changes locally before submitting.
- Document new features or changes in the README.

**Known Issues**:

- Second image upload in `predict.html` may fail due to form state persistence. Debugging logs (console, network, server) are needed to resolve this.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- **Dataset**: PlantVillage for providing the leaf image dataset.
- **Libraries**: TensorFlow, Flask, NumPy, Pillow, and Werkzeug.
- **Inspiration**: Academic research in AI for agriculture.

---

For questions or issues, open an issue on GitHub or contact the project maintainers. Happy researching!
