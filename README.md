
# AI-Based Driver Drowsiness Detection System

## 1. Project Overview

The AI-Based Driver Drowsiness Detection System is a computer vision project designed to monitor driver alertness using a webcam.

The system detects facial landmarks, monitors eye closure, identifies yawning and generates an audio alarm when prolonged eye closure is detected.

## 2. Objectives

- Detect the driver's face in real time.
- Identify open and closed eyes.
- Calculate Eye Aspect Ratio (EAR).
- Calculate Mouth Aspect Ratio (MAR).
- Detect yawning using a CNN model.
- Identify prolonged eye closure.
- Generate an audio alert for possible drowsiness.

## 3. Technologies Used

- Python
- TensorFlow and Keras
- OpenCV
- MediaPipe
- NumPy
- Tkinter
- Scikit-learn

## 4. Algorithms Used

### Convolutional Neural Network (CNN)
Used for eye-state and yawn classification.

### Eye Aspect Ratio (EAR)
Used to estimate whether the eyes are open or closed using facial landmarks.

### Mouth Aspect Ratio (MAR)
Used to measure mouth opening and support yawn detection.

### MediaPipe Face Mesh
Used to identify facial landmarks.

## 5. Project Structure

```text
driver drowsiness detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── eye_classifier.keras
│   └── yawn_classifier_v2.keras
│
└── src/
    ├── final_drowsiness_system.py
    ├── test_system.py
    ├── train_eye_model.py
    ├── train_yawn_model_v2.py
    ├── evaluate_eye_model.py
    ├── evaluate_yawn_v2.py
    ├── face_detection.py
    ├── facial_landmarks.py
    ├── eye_aspect_ratio.py
    └── mouth_aspect_ratio.py
```

## 6. How the System Works

1. The webcam captures the driver's face.
2. MediaPipe identifies facial landmarks.
3. EAR is calculated to monitor eye closure.
4. MAR is calculated to measure mouth opening.
5. CNN models predict eye state and yawn probability.
6. The system monitors prolonged eye closure.
7. An audio alarm is generated when the eye-closure condition is met.

## 7. Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd "driver drowsiness detection"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 8. Run the Application

Start the desktop application:

```bash
python app.py
```

Click START DETECTION to launch webcam monitoring.

## 9. Model Performance

Eye CNN test accuracy: 100% on the current test split.

Yawn CNN V2 test accuracy: 86.76% on the current test split.

Note: These results are from the existing dataset split and do not establish real-world performance. Independent testing is required.

## 10. Future Improvements

- Improve accuracy using more diverse datasets.
- Reduce false-positive predictions.
- Add head-pose estimation.
- Add driver fatigue analysis.
- Develop a real-time monitoring dashboard.
- Test under different lighting conditions.

## 11. Disclaimer

This project is developed for educational and research purposes. It is not a certified driver safety system and should not be relied upon while driving.
