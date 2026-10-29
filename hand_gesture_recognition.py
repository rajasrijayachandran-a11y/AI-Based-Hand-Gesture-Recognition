"""
AI-Based Hand Gesture Recognition System
Recognizes a few predefined hand gestures using MediaPipe hand landmarks.
Run locally with a webcam: python hand_gesture_recognition.py
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def recognize_gesture(hand_landmarks):
    """Simple landmark-based gesture classifier."""
    lm = hand_landmarks.landmark

    # Finger tips and joints: index, middle, ring, pinky
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers_up = [lm[tip].y < lm[pip].y for tip, pip in zip(tips, pips)]

    # Thumb is approximated using horizontal distance from the index MCP.
    thumb_up = abs(lm[4].x - lm[2].x) > abs(lm[3].x - lm[2].x)
    count = sum(fingers_up)

    if count == 4 and thumb_up:
        return "OPEN PALM"
    if count == 0 and not thumb_up:
        return "FIST"
    if fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
        return "PEACE"
    if fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
        return "ONE"
    if count == 3:
        return "THREE"
    return "UNKNOWN"

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found. Run this script on a computer with a webcam.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            label = "NO HAND DETECTED"

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                label = recognize_gesture(hand)

            cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("AI Hand Gesture Recognition - Press Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
