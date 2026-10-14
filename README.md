# Smart Car Accident Detection — Prototype

## Run on Windows
1. Install Python 3.10–3.13 if needed.
2. Open this folder in VS Code.
3. In Terminal run:
   `py -m pip install -r requirements.txt`
4. Then run:
   `py app.py`
5. Open `http://127.0.0.1:5000` in your browser.

## Important limitation
This is a working video-upload and OpenCV motion/frame-change screening prototype. It is **not a trained accident-detection AI model**. It can miss accidents or flag unrelated motion. It does not provide GPS, SMS, or emergency alerts. Do not use it for real-world emergency decisions. For real accident classification, train/evaluate a model on a labelled accident/non-accident dataset and integrate it into `analyze_video()`.
