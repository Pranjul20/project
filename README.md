# OpenCV Template Object Detection & Tracking (ORB + Homography)

This project detects and tracks a given reference object (template image) in a live camera feed or a video file using **classic OpenCV techniques** (no deep learning).

The system uses **ORB feature detection + feature matching + homography estimation** to locate the object and draw a bounding box around it in real time.

---

Features
- Detects a reference object using ORB keypoints
- Matches features using BFMatcher (KNN Matching + Lowe’s Ratio Test)
- Estimates object position using Homography (RANSAC)
- Draws bounding polygon bounding box around detected object
- Displays match percentage as confidence score
- Works with both:
  - Live webcam feed
  - Pre-recorded video file

---

Requirements
Install dependencies:

```bash
pip install opencv-python numpy
```
----



Key Design Decisions
1. ORB Feature Detection (No Deep Learning)

ORB (Oriented FAST and Rotated BRIEF) was chosen because:

It is fast and efficient for real-time tracking

Works well without GPU support

Suitable for classic CV based assignments

2. BFMatcher with Lowe’s Ratio Test

Instead of direct matching, KNN matching with Lowe’s ratio test was used for robustness.
This reduces false matches and improves stability.

3. Homography + RANSAC for Object Localization

Homography transformation is used to map the template object into the live video frame.
RANSAC helps reject outlier matches and improves tracking reliability.

4. Confidence Score (Recognition Percentage)

A simple confidence percentage is calculated based on the number of good matches.

----

Known Limitations

Works best when the template image has high texture / rich features
(plain objects may fail due to lack of keypoints).

Performance decreases under:

low lighting

motion blur

heavy occlusion

Matching may become unstable if the object becomes very small in the frame.

Background clutter can produce false matches in complex scenes.

-----


Improvements (If Given More Time)

If more time was available, the following could improve the system:

Add temporal smoothing to reduce bounding box jitter

Use FLANN matcher for faster matching

Add adaptive thresholding for match count instead of fixed values

Add object tracking fallback (like CSRT/KCF) after first detection

Add automatic rejection if homography is unstable (bad polygon distortion)

Optimize by limiting feature search to region of interest (ROI)
