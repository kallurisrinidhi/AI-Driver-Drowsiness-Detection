
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import winsound
import threading
import time
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
EYE_MODEL_PATH = PROJECT_DIR / "models" / "eye_classifier.keras"
YAWN_MODEL_PATH = PROJECT_DIR / "models" / "yawn_classifier_v2.keras"

# Load models
eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH)

# MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.60
YAWN_THRESHOLD = 0.40

EYE_CLOSED_TIME = 1.5
YAWN_WARNING_TIME = 1.5
ALARM_COOLDOWN = 3

eye_closed_start = None
yawn_start = None
last_alarm_time = 0
alarm_active = False

# Calculate distance
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Calculate Eye Aspect Ratio
def calculate_ear(landmarks, indices, width, height):
    points = []

    for index in indices:
        landmark = landmarks[index]
        points.append((landmark.x * width, landmark.y * height))

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    return (vertical1 + vertical2) / (2.0 * horizontal)


# Calculate Mouth Aspect Ratio
def calculate_mar(landmarks, width, height):
    indices = [13, 14, 78, 308, 82, 312]
    points = []

    for index in indices:
        landmark = landmarks[index]
        points.append((landmark.x * width, landmark.y * height))

    vertical1 = distance(points[0], points[1])
    vertical2 = distance(points[4], points[5])
    horizontal = distance(points[2], points[3])

    return (vertical1 + vertical2) / (2.0 * horizontal)


# Play alarm
def play_alarm():
    global alarm_active

    alarm_active = True

    try:
        for _ in range(3):
            winsound.Beep(2000, 500)
            time.sleep(0.2)
    finally:
        alarm_active = False


# Trigger alarm
def trigger_alarm():
    global last_alarm_time

    current_time = time.time()

    if not alarm_active and current_time - last_alarm_time >= ALARM_COOLDOWN:
        last_alarm_time = current_time
        threading.Thread(target=play_alarm, daemon=True).start()


# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam not detected.")
    raise SystemExit

print("Driver Drowsiness Detection Started")
print("Press Q to exit")

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to read webcam.")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status = "NORMAL"
    status_color = (0, 255, 0)

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        # Calculate EAR
        left_ear = calculate_ear(
            landmarks,
            [362, 385, 387, 263, 373, 380],
            width,
            height
        )

        right_ear = calculate_ear(
            landmarks,
            [33, 160, 158, 133, 153, 144],
            width,
            height
        )

        ear = (left_ear + right_ear) / 2.0

        # Calculate MAR
        mar = calculate_mar(landmarks, width, height)

        # Eye CNN prediction
        eye_y1 = int(height * 0.20)
        eye_y2 = int(height * 0.65)
        eye_x1 = int(width * 0.15)
        eye_x2 = int(width * 0.85)

        eye_crop = frame[eye_y1:eye_y2, eye_x1:eye_x2]

        if eye_crop.size > 0:
            eye_input = cv2.resize(eye_crop, (64, 64))
            eye_input = np.expand_dims(eye_input, axis=0)

            eye_prediction = float(
                eye_model.predict(eye_input, verbose=0)[0][0]
            )
        else:
            eye_prediction = 1.0

        # Yawn CNN prediction
        yawn_input = cv2.resize(frame, (64, 64))
        yawn_input = np.expand_dims(yawn_input, axis=0)

        yawn_probability = float(
            yawn_model.predict(yawn_input, verbose=0)[0][0]
        )

        # Display YAWN or NO YAWN
        if yawn_probability >= YAWN_THRESHOLD:
            yawn_label = "YAWN"
            yawn_color = (0, 0, 255)
        else:
            yawn_label = "NO YAWN"
            yawn_color = (0, 255, 0)

        # Eye detection using EAR
        eye_closed = ear < EAR_THRESHOLD

        if eye_closed:
            if eye_closed_start is None:
                eye_closed_start = time.time()
        else:
            eye_closed_start = None

        # Yawn detection using MAR and CNN
        possible_yawn = (
            mar > MAR_THRESHOLD
            and yawn_probability >= YAWN_THRESHOLD
        )

        if possible_yawn:
            if yawn_start is None:
                yawn_start = time.time()
        else:
            yawn_start = None

        # Eye closure alarm
        if eye_closed_start is not None:
            if time.time() - eye_closed_start >= EYE_CLOSED_TIME:
                status = "DROWSINESS ALERT!"
                status_color = (0, 0, 255)
                trigger_alarm()

        # Sustained yawn warning
        if yawn_start is not None:
            if time.time() - yawn_start >= YAWN_WARNING_TIME:
                if status == "NORMAL":
                    status = "YAWNING WARNING"
                    status_color = (0, 165, 255)

        # Eye status
        eye_status = "CLOSED" if eye_closed else "OPEN"

        # Display EAR
        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Display MAR
        cv2.putText(
            frame,
            f"MAR: {mar:.3f}",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Display eye status
        cv2.putText(
            frame,
            f"Eyes: {eye_status}",
            (25, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Display yawn probability
        cv2.putText(
            frame,
            f"Yawn probability: {yawn_probability:.2f}",
            (25, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # Display eye CNN probability
        cv2.putText(
            frame,
            f"Eye CNN probability: {eye_prediction:.2f}",
            (25, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # Display YAWN / NO YAWN
        cv2.putText(
            frame,
            f"Yawn Status: {yawn_label}",
            (25, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            yawn_color,
            2
        )

    else:
        eye_closed_start = None
        yawn_start = None

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    # Display overall system status
    cv2.putText(
        frame,
        status,
        (25, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        3
    )

    cv2.imshow(
        "Final Driver Drowsiness Detection System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
camera.release()
face_mesh.close()
cv2.destroyAllWindows()

print("Detection stopped.")
