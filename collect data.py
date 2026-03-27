import os
import cv2
import pickle
import mediapipe as mp

dataset_dir = './data'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

number_of_classes = input("How many signs do you want to recognize? [default: 3]: ")
try:
    number_of_classes = int(number_of_classes)
except ValueError:
    number_of_classes = 3

dataset_size = 100

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

data = []
labels = []

for j in range(number_of_classes):
    print(f"Getting ready for class {j}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f'Ready for class {j}? Press "s" to start!', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('s'):
            break

    print(f"Collecting frames for class {j}...")
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
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
                    
                data.append(landmark_list)
                labels.append(j)
                counter += 1
                if counter >= dataset_size:
                    break

cap.release()
cv2.destroyAllWindows()

print(f"Dataset collected. Total samples: {len(data)}")
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("Saved to data.pickle!")
