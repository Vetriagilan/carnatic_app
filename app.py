from flask import Flask, request, jsonify, render_template
import os
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
import librosa
import numpy as np
import joblib
from werkzeug.utils import secure_filename
import gc # Added for memory management

app = Flask(__name__)

# Load your pre-trained model and scaler
svm_model = joblib.load('carnatic_svm_model.pkl')
scaler = joblib.load('carnatic_scaler.pkl')

os.makedirs("temp_audio", exist_ok=True)

def extract_carnatic_features(file_path):
    try:
        # OPTIMIZATION: sr=16000 slashes memory usage in half for Render's free tier
        y, sr = librosa.load(file_path, sr=16000, duration=30)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        
        # Explicitly free up memory immediately
        del y
        gc.collect()
        
        return np.hstack([mfccs_mean, chroma_mean, contrast_mean])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join("temp_audio", filename)
    file.save(filepath)
    
    features = extract_carnatic_features(filepath)
    if features is None:
        os.remove(filepath)
        return jsonify({'error': 'Failed to extract features from audio'}), 500
        
    features_scaled = scaler.transform(features.reshape(1, -1))
    probs = svm_model.predict_proba(features_scaled)[0]
    
    classes = svm_model.classes_ 
    prob_dict = {cls.lower(): float(prob) for cls, prob in zip(classes, probs)}
    
    predicted_idx = np.argmax(probs)
    predicted_emotion = classes[predicted_idx].lower()
    
    os.remove(filepath)
    
    return jsonify({
        'emotion': predicted_emotion,
        'probs': prob_dict
    })

# RENDER PORT BINDING FIX
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)