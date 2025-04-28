from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import markdown
import google.generativeai as genai
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

GEMINI_API_KEY = "REPLACE WITH YOUR API KEY"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction="You are a friendly and knowledgeable gardening expert named Raju. "
        "introduce yourself only when asked and not anywhere else. "
        "You specialize in plant care, soil science, pest control, and organic gardening. "
        "You will always respond as Raju, a virtual gardening assistant. "
        "Never mention that you are an AI language model. "
        "Always answer as a gardening specialist, not a general AI. "
        "act like a human. "
        "only answer gardening or crop or plants related questions")

class Cast(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

MOBILENET_MODEL_PATH = "best_disease_model.keras"
mobilenet_model = tf.keras.models.load_model(MOBILENET_MODEL_PATH, custom_objects={"Cast": Cast})

EFFICIENTNET_MODEL_PATH = "best_disease_model_b3.keras"
efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_MODEL_PATH, custom_objects={"Cast": Cast})

MOBILENET_IMG_SIZE = (224, 224)
EFFICIENTNET_IMG_SIZE = (300, 300)

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def preprocess_image_mobilenet(img_path):
    img = image.load_img(img_path, target_size=MOBILENET_IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def preprocess_image_efficientnet(img_path):
    img = image.load_img(img_path, target_size=EFFICIENTNET_IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def classify_image_mobilenet(img_path):
    if not os.path.exists(img_path):
        return "Error: Image file not found!", 0.0
    
    img_array = preprocess_image_mobilenet(img_path)
    predictions = mobilenet_model.predict(img_array, verbose=0)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]
    
    return CLASS_NAMES[class_index], confidence

def classify_image_efficientnet(img_path):
    if not os.path.exists(img_path):
        return "Error: Image file not found!", 0.0
    
    img_array = preprocess_image_efficientnet(img_path)
    predictions = efficientnet_model.predict(img_array, verbose=0)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]
    
    return CLASS_NAMES[class_index], confidence

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/bot')
def bot():
    prompt = request.args.get('prompt', '')
    return render_template('chatbot.html', initial_prompt=prompt)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"]
    try:
        response = gemini_model.generate_content(user_message)
        bot_reply = markdown.markdown(response.text) if response.text else "I'm not sure how to respond to that."
        return jsonify({"response": bot_reply})
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            if mobilenet_model is None:
                return jsonify({"error": "Error: MobileNetV2 model not loaded. Please try again later."}), 500
            
            predicted_class, confidence = classify_image_mobilenet(file_path)
            
            image_url = f"/static/uploads/{filename}"
            is_healthy = "healthy" in predicted_class.lower()
            
            return render_template('predict1.html', 
                                 image_path=image_url,
                                 name=f"Prediction (MobileNetV2): {predicted_class} (Confidence: {confidence:.2f})",
                                 is_healthy=is_healthy,
                                 disease_name=predicted_class)
        else:
            return jsonify({"error": "Invalid file type. Please upload an image file (png, jpg, jpeg, gif)."}), 400
    
    return render_template('predict1.html')

@app.route('/predict2', methods=['GET', 'POST'])
def predict2():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            if efficientnet_model is None:
                return jsonify({"error": "Error: EfficientNetB3 model not loaded. Please try again later."}), 500
            
            predicted_class, confidence = classify_image_efficientnet(file_path)
            
            image_url = f"/static/uploads/{filename}"
            is_healthy = "healthy" in predicted_class.lower()
            
            return render_template('predict2.html', 
                                 image_path=image_url,
                                 name=f"Prediction (EfficientNetB3): {predicted_class} (Confidence: {confidence:.2f})",
                                 is_healthy=is_healthy,
                                 disease_name=predicted_class)
        else:
            return jsonify({"error": "Invalid file type. Please upload an image file (png, jpg, jpeg, gif)."}), 400
    
    return render_template('predict2.html')

@app.route('/get_prevention_cure', methods=['POST'])
def get_prevention_cure():
    data = request.get_json()
    if not data or "disease_name" not in data:
        return jsonify({"error": "No disease name provided"}), 400

    disease_name = data["disease_name"]
    try:
        if "healthy" in disease_name.lower():
            return jsonify({"gemini_response": "Crop is healthy"})
        
        response = gemini_model.generate_content(
            f"Provide prevention and cure methods for {disease_name}. Use proper bullet points and bold formatting."
        )
        formatted_response = markdown.markdown(response.text)
        return jsonify({"gemini_response": formatted_response})
    except Exception as e:
        return jsonify({"error": f"Error fetching response: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
