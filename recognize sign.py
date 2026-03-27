import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
try:
    with open('model.p', 'rb') as f:
        model_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: model.p not found. Please run 2_train_model.py first.")
    exit()

model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Dictionary for labels mapping (full alphabet A-Z)
labels_dict = {i: chr(65 + i) for i in range(26)}

print("Starting real-time recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            landmark_list = []
            x_ = []
            y_ = []
            
            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)
                
            x_min, y_min = min(x_), min(y_)
            x_max, y_max = max(x_), max(y_)
            
            scale = max(x_max - x_min, y_max - y_min, 1e-5)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                landmark_list.append((x - x_min) / scale)
                landmark_list.append((y - y_min) / scale)
                
            # Make prediction
            prediction = model.predict([np.asarray(landmark_list)])
            predicted_class = prediction[0]
            
            # Use label dictionary or fallback to string format of predicted class
            predicted_character = labels_dict.get(int(predicted_class), str(predicted_class))
            
            # Display prediction
            # Draw bounding box roughly based on min/max points to place text
            x_coord = int(x_min * frame.shape[1]) - 20
            y_coord = int(y_min * frame.shape[0]) - 20
            
            cv2.putText(frame, f'Sign: {predicted_character}', (max(0, x_coord), max(20, y_coord)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
