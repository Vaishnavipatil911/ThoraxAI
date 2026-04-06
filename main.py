import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np


st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .pneumonia-detected {
        background-color: #ffebee;
        border: 3px solid #f44336;
        color: #c62828;
    }
    .normal-lungs {
        background-color: #e8f5e9;
        border: 3px solid #4caf50;
        color: #2e7d32;
    }
    .confidence-text {
        font-size: 1.2rem;
        margin-top: 1rem;
        color: #555;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .debug-info {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with st.spinner("Loading AI model..."):
        try:
            
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
          
            model_options = [
                "dima806/chest_xray_pneumonia_classification",
                "nickmuchi/vit-finetuned-chest-xray-pneumonia",
                "edumunozsala/vit-base-patch16-224-in21k-finetuned-chest-xray-pneumonia"
            ]
            
            for model_name in model_options:
                try:
                    processor = AutoImageProcessor.from_pretrained(model_name)
                    model = AutoModelForImageClassification.from_pretrained(model_name)
                    st.success(f"✅ Loaded model: {model_name}")
                    return processor, model, "huggingface", model_name
                except:
                    continue
            
            
            st.warning("⚠️ Using fallback model - results may vary")
            return None, None, "fallback", None
            
        except Exception as e:
            st.error(f"Model loading error: {e}")
            return None, None, "fallback", None

processor, model, model_type, model_name = load_model()

def predict_with_huggingface(img: Image.Image):
    """Predict using HuggingFace model"""
    inputs = processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    
    predicted_class = logits.argmax(-1).item()
    confidence = probs[0][predicted_class].item()
  
    id2label = model.config.id2label
    label = id2label[predicted_class]
    
   
    if "pneumonia" in label.lower():
        return "Pneumonia", confidence
    else:
        return "Normal", confidence

def predict_with_fallback(img: Image.Image):
    """Fallback prediction based on image analysis"""

    gray_img = img.convert('L')
    img_array = np.array(gray_img)
    
    mean_intensity = img_array.mean()
    std_intensity = img_array.std()
    
    
    bright_threshold = mean_intensity + std_intensity
    bright_pixels = (img_array > bright_threshold).sum()
    total_pixels = img_array.size
    bright_ratio = bright_pixels / total_pixels
    
   
    if bright_ratio > 0.3: 
        return "Pneumonia", 0.65
    else:
        return "Normal", 0.60

st.markdown("""
<div class="main-header">
    <h1>🫁 Pneumonia Detection System</h1>
    <p>AI-Powered Chest X-ray Analysis</p>
</div>
""", unsafe_allow_html=True)

st.info("ℹ️ **Disclaimer:** This is an educational AI tool. Results should not be used for clinical diagnosis. Always consult a qualified healthcare professional.")

if model_type == "fallback":
    st.warning("⚠️ **Note:** Using fallback detection method. For best results, please ensure you have internet connectivity for model download.")


with st.expander("📋 How to Use", expanded=False):
    st.markdown("""
    1. Upload a chest X-ray image (JPG, JPEG, or PNG format)
    2. Wait for the AI model to analyze the image
    3. View the detection results and confidence score
    
    **Tips for Best Results:**
    - Use clear, high-quality chest X-ray images
    - Ensure the X-ray shows the full chest area
    - Frontal view X-rays work best
    """)


st.markdown("### 📤 Upload Chest X-ray Image")

uploaded_file = st.file_uploader(
    "Choose an X-ray image file",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear chest X-ray image for analysis"
)


if uploaded_file is not None:
    try:
        # Load and display image
        img = Image.open(uploaded_file).convert("RGB")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img, caption="Uploaded X-ray Image", use_container_width=True)
        
        # Analyze button
        st.markdown("---")
        
        with st.spinner("🔍 Analyzing X-ray image..."):
            if model_type == "huggingface" and model is not None:
                label, confidence = predict_with_huggingface(img)
            else:
                label, confidence = predict_with_fallback(img)
        
        # Display results
        st.markdown("### 📊 Detection Results")
        
        if label == "Pneumonia":
            st.markdown(f"""
            <div class="result-box pneumonia-detected">
                🔴 PNEUMONIA DETECTED
                <div class="confidence-text">Confidence: {confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.warning("⚠️ The AI model has detected signs consistent with pneumonia. Please consult a healthcare professional for proper diagnosis and treatment.")
            
        else:
            st.markdown(f"""
            <div class="result-box normal-lungs">
                🟢 NORMAL LUNGS
                <div class="confidence-text">Confidence: {confidence*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ The AI model indicates normal lung appearance. However, this does not replace professional medical evaluation.")
        
        # Debug information
        with st.expander("🔧 Technical Details"):
            st.markdown(f"""
            <div class="debug-info">
            <strong>Model Information:</strong><br>
            - Model Type: {model_type}<br>
            - Model Name: {model_name if model_name else 'Fallback method'}<br>
            - Prediction: {label}<br>
            - Confidence: {confidence*100:.2f}%<br>
            </div>
            """, unsafe_allow_html=True)
            
            if model_type == "huggingface":
                st.markdown("""
                **About This Model:**
                This model was specifically trained on chest X-ray images to detect pneumonia.
                It uses Vision Transformer (ViT) or ResNet architecture fine-tuned on medical imaging data.
                """)
            else:
                st.markdown("""
                **About Fallback Method:**
                Using basic image analysis to detect opacity patterns.
                For accurate results, please ensure stable internet connection to download the trained model.
                """)
        
        # Additional info
        with st.expander("ℹ️ Understanding the Results"):
            st.markdown(f"""
            - **Prediction:** {label}
            - **Confidence Score:** {confidence*100:.1f}%
            
            **What This Means:**
            - The confidence score indicates how certain the AI model is about its prediction
            - Scores above 80% suggest higher certainty
            - Scores below 60% suggest the model is uncertain
            
            **Important Notes:**
            - This tool is for educational purposes only
            - AI predictions should never replace professional medical diagnosis
            - Chest X-rays should always be interpreted by qualified radiologists
            - If you have symptoms, please consult a healthcare provider
            """)
            
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please try uploading a different image or check that the file is a valid X-ray image.")

else:
    st.markdown("""
    <div class="upload-section">
        <h3>👆 Upload an X-ray image to begin analysis</h3>
        <p>Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("🔬 Powered by PyTorch & HuggingFace Transformers | For Educational Purposes Only")

# Test with sample data button
if st.button("🧪 Need test X-ray images?"):
    st.info("""
    **You can find sample chest X-ray images here:**
    - Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    - GitHub: Search for "chest xray pneumonia dataset"
    
    Make sure to use images from the test set for evaluation.
    """)