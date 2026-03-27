import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading data...")
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: data.pickle not found. Please run 1_collect_data.py first.")
    exit()

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Loaded {len(data)} samples.")

if len(data) == 0:
    print("No data collected. Exiting.")
    exit()

# Split the dataset for training and testing
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# We use a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)

print("Training model...")
model.fit(x_train, y_train)

# Test the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"Model accuracy on test set: {score * 100:.2f}%")

# Save the trained model
print("Saving model to model.p...")
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("Model Saved! You can now run 3_recognize_sign.py")
