
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Eye landmark points
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Temporary threshold for testing
EAR_THRESHOLD = 0.30


def calculate_ear(landmarks, eye_points, width, height):

    points = []

    for index in eye_points:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)
        points.append([x, y])

    points = np.array(points)

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

print("Eye detection started. Press Q to exit.")

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
                print("Unable to read camera.")
                break

            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(rgb_frame)

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

                    average_ear = (
                        left_ear + right_ear
                    ) / 2.0

                    if average_ear < EAR_THRESHOLD:
                        eye_status = "CLOSED"
                        text_color = (0, 0, 255)
                    else:
                        eye_status = "OPEN"
                        text_color = (0, 255, 0)

                    cv2.putText(
                        frame,
                        f"EAR: {average_ear:.3f}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Eye Status: {eye_status}",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        text_color,
                        2
                    )

            else:
                cv2.putText(
                    frame,
                    "Face not detected",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Driver Drowsiness - Eye Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Camera closed successfully.")
