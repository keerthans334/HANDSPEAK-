import os
import cv2
import mediapipe as mp
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from landmark_detection import collect_landmarks

# Gesture names with potential special characters
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

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Function to sanitize gesture names (to remove special characters)
def sanitize_gesture_name(gesture_name):
    return gesture_name.replace(",", "").replace("?", "").replace(" ", "_")

# Create gesture folders with sanitized names
def create_folders():
    os.makedirs("gesture_images", exist_ok=True)
    for gesture in GESTURES:
        sanitized_gesture = sanitize_gesture_name(gesture)
        os.makedirs(os.path.join("gesture_images", sanitized_gesture), exist_ok=True)

# Capture 50 images for each gesture
def capture_images():
    cap = cv2.VideoCapture(0)
    print("Press a number (0-7) to save gesture image, 'q' to quit.")
    
    # Create image count dictionary with sanitized folder names
    image_count = {sanitize_gesture_name(gesture): len(os.listdir(os.path.join("gesture_images", sanitize_gesture_name(gesture)))) for gesture in GESTURES}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Show landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Capture Gestures", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord(str(i)) for i in range(len(GESTURES))]:
            gesture_idx = int(chr(key))
            gesture_name = GESTURES[gesture_idx]
            sanitized_gesture_name = sanitize_gesture_name(gesture_name)
            folder_path = os.path.join("gesture_images", sanitized_gesture_name)

            # Check if we've already collected 50 images for this gesture
            if image_count[sanitized_gesture_name] >= 50:
                print(f"Already collected 50 images for '{gesture_name}'.")
                continue

            img_path = os.path.join(folder_path, f"{sanitized_gesture_name}_{image_count[sanitized_gesture_name]}.jpg")
            cv2.imwrite(img_path, frame)
            image_count[sanitized_gesture_name] += 1
            print(f"Saved: {img_path}")

            # Notify if 50 images are complete
            if image_count[sanitized_gesture_name] == 50:
                print(f"Collected 50 images for '{gesture_name}'.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Train model
def train_model():
    data, labels = [], []

    # Ensure we train only on gestures with valid data
    for idx, gesture in enumerate(GESTURES):
        sanitized_gesture_name = sanitize_gesture_name(gesture)
        folder_path = os.path.join("gesture_images", sanitized_gesture_name)
        
        # Skip empty folders
        if not os.path.exists(folder_path) or len(os.listdir(folder_path)) == 0:
            print(f"Skipping '{gesture}' as the folder is empty.")
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            print(f"Loading image from: {img_path}")  # Debugging line
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue  # Skip this image and move to the next one

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            landmarks = collect_landmarks(img, results)
            if landmarks:
                data.append(landmarks)
                labels.append(idx)

    # Check if there is enough data to train
    if len(data) == 0:
        print("No data available for training. Ensure gesture images are captured.")
        return

    # Train and evaluate models
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[sanitize_gesture_name(gesture) for gesture in GESTURES]))

    print(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

    # Save the model
    with open("gesture_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved as gesture_model.pkl.")

if __name__ == "__main__":
    create_folders()  # Create sanitized folders for gestures
    capture_images()  # Capture gesture images (up to 50 per gesture)
    train_model()     # Train the model with captured data
