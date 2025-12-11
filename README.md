🎭 AI-Powered Deepfake Video Detection

A deep learning–based system that analyzes videos, extracts frames, and predicts whether the content is real or manipulated using a custom-trained model. Built for accuracy, robustness, and fast inference.

✨ Core Features

🎬 Smart Frame Extraction: Extracts key frames for more reliable prediction.

🧠 Deepfake Classification Model: CNN/Transformer-based architecture trained on real vs fake datasets.

📊 Confidence Scores: Shows how likely the video is FAKE or REAL.

🖼️ Visual Frame Output: Displays sample frames used for prediction.

⚡ Optimized Pipeline: Fast preprocessing + embedding extraction.

🔍 Optional Explainability (XAI): Simple text-based explanation support.

🚀 Tech Stack
ML + Backend


https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white


https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white


https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white


https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white


https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white


https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white

Tools


https://img.shields.io/badge/FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white


https://img.shields.io/badge/XAI-000000?style=for-the-badge&logoColor=white

🧠 How It Works

User uploads a video (MP4, AVI, etc.)

Frames are extracted at fixed intervals.

Each frame is preprocessed and passed to the ML model.

The model outputs REAL vs FAKE probabilities.

Multiple frame predictions are aggregated into a final decision.

(Optional) XAI module generates simple explanations.
