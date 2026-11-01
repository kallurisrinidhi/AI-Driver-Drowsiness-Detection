
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Mouth landmark points
MOUTH_POINTS = [13, 14, 78, 308, 82, 312]

def calculate_mar(landmarks, width, height):

    points = []

    for index in MOUTH_POINTS:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)
        points.append([x, y])

    points = np.array(points)

    vertical_1 = np.linalg.norm(points[0] - points[1])
    vertical_2 = np.linalg.norm(points[4] - points[5])
    horizontal = np.linalg.norm(points[2] - points[3])

    if horizontal == 0:
        return 0.0

    mar = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return mar


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

print("Mouth detection started. Press Q to exit.")

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

            if results.multi_face_landmarks:

                for face in results.multi_face_landmarks:

                    mar = calculate_mar(
                        face.landmark,
                        width,
                        height
                    )

                    cv2.putText(
                        frame,
                        f"MAR: {mar:.3f}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
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

            cv2.imshow("Mouth Aspect Ratio Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Camera closed successfully.")
