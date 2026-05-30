from flask import Flask, request, jsonify, render_template
import os
import librosa
import numpy as np
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your pre-trained model and scaler
svm_model = joblib.load('carnatic_svm_model.pkl')
scaler = joblib.load('carnatic_scaler.pkl')

# Create a temporary folder for uploaded audio
os.makedirs("temp_audio", exist_ok=True)

# ---------------------------------------------------------
# Your Exact Feature Extraction Function
# ---------------------------------------------------------
def extract_carnatic_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        
        return np.hstack([mfccs_mean, chroma_mean, contrast_mean])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ---------------------------------------------------------
# API Routes
# ---------------------------------------------------------
@app.route('/')
def home():
    # Serves your beautiful HTML UI
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Save the file temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join("temp_audio", filename)
    file.save(filepath)
    
    # Extract features
    features = extract_carnatic_features(filepath)
    if features is None:
        os.remove(filepath)
        return jsonify({'error': 'Failed to extract features from audio'}), 500
        
    # Predict
    features_scaled = scaler.transform(features.reshape(1, -1))
    probs = svm_model.predict_proba(features_scaled)[0]
    
    # Map probabilities to class names (formatting them to lowercase for the JS to read)
    classes = svm_model.classes_ 
    prob_dict = {cls.lower(): float(prob) for cls, prob in zip(classes, probs)}
    
    predicted_idx = np.argmax(probs)
    predicted_emotion = classes[predicted_idx].lower()
    
    # Cleanup temp file
    os.remove(filepath)
    
    # Send data back to the HTML UI
    return jsonify({
        'emotion': predicted_emotion,
        'probs': prob_dict
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)