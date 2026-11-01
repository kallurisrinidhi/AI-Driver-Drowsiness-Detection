
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import csv
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

EYE_MODEL_PATH = PROJECT_DIR / "models" / "eye_classifier.keras"
YAWN_MODEL_PATH = PROJECT_DIR / "models" / "yawn_classifier_v2.keras"

# Load models
eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH)

# Face detection
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create results file
results_file = PROJECT_DIR / "testing_results.csv"

file = open(results_file, "w", newline="")
writer = csv.writer(file)

writer.writerow([
    "Time",
    "EAR",
    "MAR",
    "Eye CNN Probability",
    "Yawn Probability",
    "Eye Status",
    "Yawn Status"
])

# Distance function
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# EAR calculation
def calculate_ear(landmarks, indices, width, height):
    points = []

    for index in indices:
        landmark = landmarks[index]
        points.append((landmark.x * width, landmark.y * height))

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    return (vertical1 + vertical2) / (2 * horizontal)


# MAR calculation
def calculate_mar(landmarks, width, height):
    indices = [13, 14, 78, 308, 82, 312]
    points = []

    for index in indices:
        landmark = landmarks[index]
        points.append((landmark.x * width, landmark.y * height))

    vertical1 = distance(points[0], points[1])
    vertical2 = distance(points[4], points[5])
    horizontal = distance(points[2], points[3])

    return (vertical1 + vertical2) / (2 * horizontal)


# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam not detected.")
    file.close()
    raise SystemExit

print("Testing started. Press Q to stop.")

last_save_time = 0

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = calculate_ear(
                landmarks, [362, 385, 387, 263, 373, 380],
                width, height
            )

            right_ear = calculate_ear(
                landmarks, [33, 160, 158, 133, 153, 144],
                width, height
            )

            ear = (left_ear + right_ear) / 2

            mar = calculate_mar(landmarks, width, height)

            # CNN predictions
            eye_crop = frame[
                int(height * 0.20):int(height * 0.65),
                int(width * 0.15):int(width * 0.85)
            ]

            if eye_crop.size > 0:
                eye_input = cv2.resize(eye_crop, (64, 64))
                eye_input = np.expand_dims(eye_input, axis=0)

                eye_probability = float(
                    eye_model.predict(eye_input, verbose=0)[0][0]
                )
            else:
                eye_probability = 0.0

            yawn_input = cv2.resize(frame, (64, 64))
            yawn_input = np.expand_dims(yawn_input, axis=0)

            yawn_probability = float(
                yawn_model.predict(yawn_input, verbose=0)[0][0]
            )

            eye_status = "CLOSED" if ear < 0.25 else "OPEN"

            yawn_status = (
                "YAWN" if yawn_probability >= 0.40
                else "NO YAWN"
            )

            # Save one record per second
            current_time = time.time()

            if current_time - last_save_time >= 1:
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    round(ear, 3),
                    round(mar, 3),
                    round(eye_probability, 3),
                    round(yawn_probability, 3),
                    eye_status,
                    yawn_status
                ])

                file.flush()
                last_save_time = current_time

            # Display values
            cv2.putText(
                frame, f"EAR: {ear:.3f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2
            )

            cv2.putText(
                frame, f"MAR: {mar:.3f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2
            )

            cv2.putText(
                frame, f"Eyes: {eye_status}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"Yawn: {yawn_status}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if yawn_status == "YAWN"
                else (0, 255, 0), 2
            )

        else:
            cv2.putText(
                frame, "NO FACE DETECTED", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2
            )

        cv2.imshow("System Testing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    face_mesh.close()
    file.close()
    cv2.destroyAllWindows()

print("Testing completed.")
print("Results saved to:", results_file)
