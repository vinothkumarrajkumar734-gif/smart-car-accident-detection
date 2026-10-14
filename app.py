import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_video(path):
    """
    Prototype screening only. Uses abrupt frame-change and motion heuristics.
    It is NOT a trained accident-recognition AI model and must not be used
    for emergency response or safety decisions.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"status": "Could not read video", "details": "Try another MP4/AVI/MOV file.", "frames_checked": 0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    sample_every = max(1, int(fps // 2) if fps > 0 else 10)  # sample about twice per second
    prev_gray = None
    frame_idx = 0
    checked = 0
    changes = []
    motion_values = []

    while checked < 240:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            frame = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                changes.append(float(np.mean(diff)))
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0)
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_values.append(float(np.mean(magnitude)))
            prev_gray = gray
            checked += 1
        frame_idx += 1
    cap.release()

    if len(changes) < 3:
        return {"status": "Not enough video frames", "details": "Upload a longer video with visible motion.", "frames_checked": checked}

    # Conservative prototype indicator: unusually large abrupt visual change
    # plus high motion in sampled frames. This cannot confirm an accident.
    change_threshold = max(22.0, float(np.mean(changes) + 2.2 * np.std(changes)))
    motion_threshold = max(4.0, float(np.mean(motion_values) + 1.5 * np.std(motion_values))) if motion_values else 4.0
    candidate = any(c >= change_threshold for c in changes) and any(m >= motion_threshold for m in motion_values)

    if candidate:
        status = "Possible incident pattern — manual review needed"
        details = "The video contains an abrupt visual change and motion spike. This is only a heuristic flag, not an accident prediction."
    else:
        status = "No strong incident pattern flagged"
        details = "The prototype did not find a strong abrupt-change pattern. This does NOT prove that the video is accident-free."

    duration = round(total_frames / fps, 1) if fps > 0 and total_frames > 0 else "unknown"
    return {
        "status": status,
        "details": details,
        "frames_checked": checked,
        "duration": duration,
        "method": "OpenCV motion / frame-change prototype (not trained AI)"
    }

@app.errorhandler(413)
def too_large(_error):
    return "File too large. Please upload a video under 100 MB.", 413

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None
    error = None
    if request.method == "POST":
        video = request.files.get("video")
        if not video or not video.filename:
            error = "Please choose a video first."
        elif not allowed_file(video.filename):
            error = "Unsupported file type. Use MP4, MOV, AVI, or MKV."
        else:
            filename = secure_filename(video.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            video.save(path)
            result = analyze_video(path)
    return render_template("index.html", result=result, filename=filename, error=error)

if __name__ == "__main__":
    app.run(debug=True)
