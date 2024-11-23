import cv2
import mediapipe as mp
import pickle
import pyttsx3  # Offline Text-to-Speech
from landmark_detection import collect_landmarks

GESTURES = [
    "Hi, how are you?",
    "Hey, can you please help me?",
    "Hey, I need some water.",
    "Hey, good job!",
    "I love you.",
    "Bye.",
    "What is your name?",
    "Yes, I agree."
]

# Initialize offline text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 180)  # Faster speech rate
engine.setProperty("volume", 0.9)  # Slightly louder speech

# Load trained model
try:
    with open("gesture_model.pkl", "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "classes_") or len(model.classes_) == 0:
        print("Model has no trained data. Please train the model with gesture images.")
        exit()
except FileNotFoundError:
    print("Model file 'gesture_model.pkl' not found. Please train the model first.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Real-time recognition with offline audio
def real_time_recognition():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS for smoother performance
    
    VERBOSE = False  # Debugging information toggle
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to optimize performance
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        landmarks = collect_landmarks(frame, results)
        
        if landmarks:
            try:
                # Predict gesture
                prediction = model.predict([landmarks])
                gesture = GESTURES[prediction[0]]
                
                # Display gesture text on video feed
                cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Speak out the gesture text using offline TTS asynchronously
                engine.say(gesture)
                engine.runAndWait()
                
                if VERBOSE:
                    print(f"Landmarks: {landmarks}")
                    print(f"Prediction: {gesture}")
            
            except Exception as e:
                if VERBOSE:
                    print(f"Prediction error: {e}")
                cv2.putText(frame, "Unknown Gesture", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    engine.stop()

if __name__ == "__main__":
    real_time_recognition()
