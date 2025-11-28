from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import tensorflow as tf
import os

app = Flask(__name__)

# Load model with error handling
try:
    model = tf.keras.models.load_model("models/my_new_plant_disease_model.keras")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

# Disease labels
label = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
         'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
         'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 
         'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
         'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
         'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
         'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
         'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
         'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
         'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Load plant disease data
try:
    with open("plant_disease.json", 'r') as file:
        plant_disease = json.load(file)
    print("✅ Plant disease data loaded successfully!")
except Exception as e:
    print(f"❌ Plant disease data loading failed: {e}")
    plant_disease = {}

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plant Disease Detection</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .upload-form { 
                border: 3px dashed #667eea; 
                padding: 40px; 
                text-align: center; 
                border-radius: 10px;
                margin: 20px 0;
            }
            .btn { 
                background: #4CAF50; 
                color: white; 
                padding: 12px 30px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
                font-size: 16px;
                margin: 10px;
            }
            .btn:hover { background: #45a049; }
            .result { 
                margin-top: 20px; 
                padding: 20px; 
                border-radius: 10px; 
                display: none;
            }
            .success { background: #d4edda; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; border: 1px solid #f5c6cb; }
            img { max-width: 300px; border-radius: 10px; margin: 10px 0; }
            h1 { color: #333; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌿 Plant Disease Detection</h1>
            <p style="text-align: center; color: #666;">Upload an image of plant leaf for disease detection</p>
            
            <div class="upload-form">
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" name="file" id="fileInput" accept=".png,.jpg,.jpeg" style="margin: 10px 0;">
                    <br>
                    <button type="submit" class="btn">🔍 Detect Disease</button>
                </form>
            </div>
            
            <div id="loading" style="display:none; text-align: center;">
                <h3>🔄 Analyzing image...</h3>
            </div>
            
            <div id="result" class="result"></div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                const loadingDiv = document.getElementById('loading');
                
                if (!fileInput.files[0]) {
                    alert('Please select an image file');
                    return;
                }
                
                loadingDiv.style.display = 'block';
                resultDiv.style.display = 'none';
                
                const formData = new FormData();
                formData.append('img', fileInput.files[0]);
                
                fetch('/upload/', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.text())
                .then(html => {
                    loadingDiv.style.display = 'none';
                    
                    // Create a temporary div to parse the HTML response
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = html;
                    
                    // Extract result from the response
                    const resultElement = tempDiv.querySelector('.result');
                    if (resultElement) {
                        resultDiv.innerHTML = resultElement.innerHTML;
                        resultDiv.style.display = 'block';
                        resultDiv.className = 'result success';
                    } else {
                        resultDiv.innerHTML = '<h3>❌ Error processing image</h3><p>Please try again with a different image.</p>';
                        resultDiv.style.display = 'block';
                        resultDiv.className = 'result error';
                    }
                })
                .catch(error => {
                    loadingDiv.style.display = 'none';
                    resultDiv.innerHTML = '<h3>❌ Error</h3><p>Failed to process image: ' + error + '</p>';
                    resultDiv.style.display = 'block';
                    resultDiv.className = 'result error';
                });
            });
        </script>
    </body>
    </html>
    '''

def extract_features(image):
    try:
        image = tf.keras.utils.load_img(image, target_size=(160, 160))
        feature = tf.keras.utils.img_to_array(image)
        feature = np.array([feature])
        return feature
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def model_predict(image):
    try:
        if model is None:
            return "Model not loaded"
            
        img = extract_features(image)
        if img is None:
            return "Error processing image"
            
        prediction = model.predict(img)
        predicted_class = label[prediction.argmax()]
        
        # Get disease info from JSON
        disease_info = plant_disease.get(predicted_class, "No information available")
        return disease_info
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Prediction failed: {str(e)}"

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        try:
            image = request.files['img']
            if image.filename == '':
                return redirect('/')
            
            # Create uploadimages folder if not exists
            if not os.path.exists('uploadimages'):
                os.makedirs('uploadimages')
                
            temp_name = f"uploadimages/temp_{uuid.uuid4().hex}_{image.filename}"
            image.save(temp_name)
            prediction = model_predict(temp_name)
            
            # Return HTML response
            return f'''
            <div class="result success">
                <h3>✅ Detection Result:</h3>
                <p><strong>Prediction:</strong> {prediction}</p>
                <img src="/uploadimages/{os.path.basename(temp_name)}" alt="Uploaded Image">
                <br>
                <a href="/" class="btn">🔄 Analyze Another Image</a>
            </div>
            '''
            
        except Exception as e:
            return f'''
            <div class="result error">
                <h3>❌ Error</h3>
                <p>Failed to process image: {str(e)}</p>
                <a href="/" class="btn">🔄 Try Again</a>
            </div>
            '''
    else:
        return redirect('/')

if __name__ == "__main__":
    print("🚀 Starting Flask server on http://localhost:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)