# AI-Based Hand Gesture Recognition System

## Project overview
This project detects a hand from a live webcam feed and recognises a small set of predefined gestures using MediaPipe hand landmarks and OpenCV.

## Gestures
- Open palm
- Fist
- Peace sign
- One finger
- Three fingers
- Unknown / no hand detected

## Tools and libraries
- Python
- OpenCV
- MediaPipe

## Setup
Install Python 3.10 or a compatible version, then run:

```bash
pip install opencv-python mediapipe
python hand_gesture_recognition.py
```

A webcam is required to run the live demo. Press `Q` to close the window.

## Methodology
1. Read a frame from the webcam.
2. Convert the frame from BGR to RGB.
3. Use MediaPipe Hands to detect hand landmarks.
4. Estimate which fingers are raised from landmark positions.
5. Map the finger pattern to a gesture label and show it on the video.

## Limitations
- This is a lightweight landmark/rule-based prototype, not a trained CNN model.
- Results may vary with hand orientation, lighting, and camera angle.
- The thumb rule is approximate and may not work equally well for both hands.

## Evaluation
Test each gesture under different lighting and orientations. Record the number of correct predictions and total attempts for each gesture to calculate accuracy:

`Accuracy = (Correct predictions / Total attempts) * 100`

## Future improvements
- Collect a labelled dataset of hand landmarks.
- Train a Random Forest or neural-network classifier.
- Add precision, recall, F1-score, and a confusion matrix.
- Add a Streamlit dashboard for uploading images or showing results.

## Project deliverables
- Source code: `hand_gesture_recognition.py`
- Documentation: this README
