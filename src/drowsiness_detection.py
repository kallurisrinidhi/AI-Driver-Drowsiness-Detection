
import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

# Eye landmark points
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmark points
MOUTH_POINTS = [13, 14, 78, 308, 82, 312]

# Initial thresholds - calibrate using your webcam
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.60

# Duration settings
EYE_CLOSED_DURATION = 1.5
YAWN_DURATION = 1.0


def calculate_ratio(landmarks, points, width, height):

    coordinates = []

    for index in points:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)
        coordinates.append([x, y])

    return np.array(coordinates)


def calculate_ear(landmarks, eye_points, width, height):

    p = calculate_ratio(
        landmarks, eye_points, width, height
    )

    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks, width, height):

    p = calculate_ratio(
        landmarks, MOUTH_POINTS, width, height
    )

    vertical_1 = np.linalg.norm(p[0] - p[1])
    vertical_2 = np.linalg.norm(p[4] - p[5])
    horizontal = np.linalg.norm(p[2] - p[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

eye_closed_start = None
yawn_start = None

print("Driver Drowsiness Detection Started")
print("Press Q to close the camera.")

try:

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:

            success, frame = camera.read()

            if not success:
                break

            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(rgb_frame)

            current_time = time.monotonic()

            if results.multi_face_landmarks:

                for face in results.multi_face_landmarks:

                    left_ear = calculate_ear(
                        face.landmark,
                        LEFT_EYE,
                        width,
                        height
                    )

                    right_ear = calculate_ear(
                        face.landmark,
                        RIGHT_EYE,
                        width,
                        height
                    )

                    ear = (left_ear + right_ear) / 2

                    mar = calculate_mar(
                        face.landmark,
                        width,
                        height
                    )

                    # Eye closure detection
                    if ear < EAR_THRESHOLD:

                        if eye_closed_start is None:
                            eye_closed_start = current_time

                        eye_duration = (
                            current_time - eye_closed_start
                        )

                        eye_status = "CLOSED"

                    else:
                        eye_closed_start = None
                        eye_duration = 0
                        eye_status = "OPEN"

                    # Yawn detection
                    if mar > MAR_THRESHOLD:

                        if yawn_start is None:
                            yawn_start = current_time

                        yawn_duration = (
                            current_time - yawn_start
                        )

                        if yawn_duration >= YAWN_DURATION:
                            mouth_status = "POSSIBLE YAWN"
                        else:
                            mouth_status = "MOUTH OPEN"

                    else:
                        yawn_start = None
                        yawn_duration = 0
                        mouth_status = "NORMAL"

                    # Display measurements
                    cv2.putText(
                        frame,
                        f"EAR: {ear:.3f} | {eye_status}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255) if eye_status == "CLOSED"
                        else (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"MAR: {mar:.3f} | {mouth_status}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255) if mouth_status == "POSSIBLE YAWN"
                        else (0, 255, 0),
                        2
                    )

                    # Drowsiness warning
                    if eye_duration >= EYE_CLOSED_DURATION:

                        warning = "WARNING: POSSIBLE DROWSINESS"

                        cv2.putText(
                            frame,
                            warning,
                            (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

            else:

                eye_closed_start = None
                yawn_start = None

                cv2.putText(
                    frame,
                    "Face not detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Driver Drowsiness Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Camera closed successfully.")
