import numpy as np

def collect_landmarks(image, results):
    """Extract landmarks from MediaPipe results."""
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [
            (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
        ]
        return np.array(landmarks).flatten()
    return None
