# OpenCV Template Object Detection & Tracking (ORB + Homography)

This project detects and tracks a given reference object (template image) in a live camera feed or a video file using **classic OpenCV techniques** (no deep learning).

The system uses **ORB feature detection + feature matching + homography estimation** to locate the object and draw a bounding box around it in real time.

---

## 📌 Features
- Detects a reference object using ORB keypoints
- Matches features using BFMatcher (KNN Matching + Lowe’s Ratio Test)
- Estimates object position using Homography (RANSAC)
- Draws bounding polygon bounding box around detected object
- Displays match percentage as confidence score
- Works with both:
  - Live webcam feed
  - Pre-recorded video file

---

## ⚙️ Requirements
Install dependencies:

```bash
pip install opencv-python numpy
