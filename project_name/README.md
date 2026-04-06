🫁 Pneumonia Detection System
AI-powered Streamlit app for detecting pneumonia from chest X-ray images using Hugging Face Transformers models.
​

[[

🚀 Quick Demo
Upload a chest X-ray to get instant pneumonia detection with confidence scores. Educational tool only—not for clinical use.

(Add a GIF of the app in action here)

✨ Features
Real-time Analysis: Upload JPG/PNG X-rays for instant AI predictions.

Multiple Models: Auto-loads fine-tuned models like dima806/chest_xray_pneumonia_classification or ViT variants.

Fallback Mode: Works offline with basic image analysis.

Visual Feedback: Color-coded results (red for pneumonia, green for normal) with confidence %.

User-Friendly: Custom CSS styling, expanders for details, and medical disclaimers.

📋 Table of Contents
Installation

Usage

Demo

Models

Limitations

Contributing

License

🛠️ Installation
Clone the repo:

text
git clone https://github.com/yourusername/pneumonia-detection-app.git
cd pneumonia-detection-app
Create a virtual environment:

text
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

text
pip install -r requirements.txt
requirements.txt:

text
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
Pillow>=10.0.0
numpy>=1.24.0
accelerate  # Optional for faster inference
🚀 Usage
Run the app locally:

text
streamlit run app.py
Opens at http://localhost:8501.

Tips:

Use high-quality frontal chest X-rays for best results.

First run downloads models (~500MB)—ensure internet.

Test with Kaggle Chest X-Ray Pneumonia dataset.

📊 Demo
Live Demo (Deploy on Streamlit Cloud for free)

🤖 Models
Prioritizes these Hugging Face models:

dima806/chest_xray_pneumonia_classification

nickmuchi/vit-finetuned-chest-xray-pneumonia

edumunozsala/vit-base-patch16-224-in21k-finetuned-chest-xray-pneumonia

Fallback: Intensity-based analysis for opacity detection.

⚠️ Limitations & Disclaimer
Educational Only: Not FDA-approved; consult radiologists for diagnosis.

Confidence <60%: Uncertain—re-upload or seek professional review.

Model biases: Trained on specific datasets; may underperform on diverse populations.

No guarantees on accuracy outside test sets.

🤝 Contributing
Fork & create a PR.

Add tests: pytest tests/.

Lint: ruff check ..

Update models or CSS welcome!

See CONTRIBUTING.md for details.
​

📄 License
MIT License. See LICENSE for details.
​

Built with ❤️ for healthcare AI prototypes. Last updated