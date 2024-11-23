# HANDSPEAK-
HANDSPEAK (Translating Signs , Transforming Lives)

#Problem Statement Description/Solution
Problem Statement :
People with speech or hearing impairments often face significant challenges in communicating with those who do not understand sign language, leading to social isolation and reduced access to opportunities in both personal and professional context.  
Solution :
Our project, HandSpeak leverages cutting-edge computer vision and machine learning to create a real-time gesture recognition system. This system captures hand gestures, interprets them into meaningful text and speech, and enables seamless communication between individuals with speech/hearing impairments and the general public.
Key Features:  
- Efficient training of gestures specific to individual needs.  
- Real-time recognition and offline text-to-speech conversion.  
- A scalable system designed to accommodate more gestures over time.  
  

#Innovation and Novelty in the work:
1. Integration of Real-Time Gesture Recognition with Offline TTS:
         - Utilized MediaPipe and Machine Learning to create a robust hand-gesture recognition model.
         - Enabled offline functionality using pyttsx3 for text-to-speech conversion, ensuring accessibility 
            without relying on internet connectivity.

2. Customizable Gesture Dataset:
         - Designed a system that allows users to train the model on specific gestures, making it adaptable 
           for various languages or use cases.

3. Interactive Training System:
          - Developed a real-time gesture image capture tool, reducing complexity for users while generating
             quality datasets for training.

4. Real-World Applicability:
           - Focused on practical usage for people with speech disabilities, bridging the gap in effective 
              communication with simple and cost-effective methods

#libraries  and functionalities that are included in this..


 1. os
   - Functionality: 
     - Provides functions to interact with the operating system.
     - Used for creating directories (`os.makedirs`) and accessing file paths (`os.path`).
   - Use in the  Code:
     - To create folders for gesture images.
     - To manage file paths for saving and loading gesture data.

 2. cv2 (OpenCV)
   - Functionality: 
     - A computer vision library for image and video processing.
     - Offers tools for real-time processing like reading, writing, and manipulating images and videos.
   - Use in the Code:
     - Captures live video frames for gesture collection (`cv2.VideoCapture`).
     - Converts image color spaces (`cv2.cvtColor`).
     - Saves gesture images (`cv2.imwrite`).
     - Displays frames and draws hand landmarks (`cv2.imshow`).

 3. mediapipe
   - Functionality:
     - A library for building machine learning pipelines, particularly for real-time face, hand, and body tracking.
   - Use in the Code:
     - Detects hand landmarks using `mediapipe.solutions.hands`.
     - Processes images to identify hand gestures and draws connections between landmarks.

 4. pickle
   - Functionality:
     - Serializes (saves) and deserializes (loads) Python objects.
   - Use in the Code:
     - Saves the trained machine learning model to a file (`gesture_model.pkl`).
     - Loads the model for future predictions.

5. sklearn (scikit-learn)
   - Functionality:
     - A machine learning library offering tools for classification, regression, clustering, and evaluation.
   - Use in the Code:
     - Splits data into training and testing sets (`train_test_split`).
     - Implements machine learning models like `RandomForestClassifier`.
     - Provides evaluation metrics such as `classification_report`.

 6. landmark_detection (Custom Module)
   - Functionality:
     - Likely contains a custom function, `collect_landmarks`, to extract hand landmark coordinates from MediaPipe results.
   - Use in the Code:
     - Converts detected hand landmarks into a usable format for model training.

 Libraries Used in Evaluation:
   - sklearn.metrics.classification_report:
     - Generates a detailed report of metrics like accuracy, precision, recall, and F1-score for model evaluation.
     - Summarizes performance across classes (gestures).
 Additional Notes:
- RandomForestClassifier:
  - A robust machine learning model for classification tasks, using ensemble decision trees.
  - Provides high accuracy and handles complex data distributions.

 #hardware componenets requirements..
 ![image](https://github.com/user-attachments/assets/f9b8d8b7-2d65-411c-8db1-500dade1e4a1)

 # what is the order to  run the project.
     # first you need to install these packages in your local system to make it run in vs code terminal
     libraries packages..
     pip install mediapipe,tensorflow,opencv,numpy,pickle,os,pyttsx3
 1. first run the landmark_detection.py to gather the landmark of your hand
 2. run the train.py to train the model using 50 images using tensor flow lite + open cv
 3. run the real_time_recognition.py to reconize the hand gestures



