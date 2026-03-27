# Sign Language Recognition System

This is a complete framework to train and run your very own sign language recognition model! It uses **OpenCV** to interact with the webcam, **MediaPipe** to extract 3D hand landmarks, and **Scikit-Learn** to do real-time classification.

## Getting Started

1. **Install Dependencies**
   Open your terminal in this directory and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Collect Data (`1_collect_data.py`)**
   ```bash
   python 1_collect_data.py
   ```
   - The script will ask you how many signs you want to recognize (default: 3). Let's say you choose `3` to train signs for `A`, `B`, and `C`.
   - It will start the camera. Once you're ready to perform your first sign (class `0`), press `s`.
   - Hold your hand steady while making the sign. It will capture 100 frames.
   - It repeats this for class `1` and class `2`.
   - Data is saved to `data.pickle`.

3. **Train the Model (`2_train_model.py`)**
   ```bash
   python 2_train_model.py
   ```
   - This script loads the points collected in step 2.
   - It trains a highly accurate Random Forest model to instantly recognize the shapes.
   - Model is saved to `model.p`.

4. **Run Real-Time Recognition (`3_recognize_sign.py`)**
   ```bash
   python 3_recognize_sign.py
   ```
   - Make the signs to the camera, and it will draw the prediction boxes!
   - Note: The script uses a `labels_dict` to map class `0` to 'A', class `1` to 'B', etc. You can freely edit this mapping in `3_recognize_sign.py` line 29.

Press `q` on the video window anytime to quit!
