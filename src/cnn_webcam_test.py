
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

print("Loading models...")

# Load trained models
eye_model = tf.keras.models.load_model(
    "models/eye_classifier.keras"
)

yawn_model = tf.keras.models.load_model(
    "models/yawn_classifier_v2.keras"
)

print("Both models loaded successfully!")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# Extract eye image
def get_eye_crop(frame, landmarks, indices, padding=30):

    height, width, _ = frame.shape

    points = []

    for index in indices:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)
        points.append([x, y])

    points = np.array(points)

    x1 = max(0, np.min(points[:, 0]) - padding)
    y1 = max(0, np.min(points[:, 1]) - padding)
    x2 = min(width, np.max(points[:, 0]) + padding)
    y2 = min(height, np.max(points[:, 1]) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


# Predict using CNN
def predict_image(model, image):

    if image is None or image.size == 0:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(image, (64, 64))

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(
        image,
        verbose=0
    )

    return float(prediction[0][0])


# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

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

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:

                for face in results.multi_face_landmarks:

                    # -------------------------
                    # EYE DETECTION
                    # -------------------------

                    left_crop = get_eye_crop(
                        frame,
                        face.landmark,
                        LEFT_EYE
                    )

                    right_crop = get_eye_crop(
                        frame,
                        face.landmark,
                        RIGHT_EYE
                    )

                    left_prob = predict_image(
                        eye_model,
                        left_crop
                    )

                    right_prob = predict_image(
                        eye_model,
                        right_crop
                    )

                    if (
                        left_prob is not None
                        and right_prob is not None
                    ):

                        eye_open_prob = (
                            left_prob + right_prob
                        ) / 2

                        eye_status = (
                            "OPEN"
                            if eye_open_prob >= 0.5
                            else "CLOSED"
                        )

                        eye_color = (
                            (0, 255, 0)
                            if eye_status == "OPEN"
                            else (0, 0, 255)
                        )

                        cv2.putText(
                            frame,
                            f"Eye: {eye_status}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            eye_color,
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Eye Open Probability: {eye_open_prob:.3f}",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 0),
                            2
                        )

                    # -------------------------
                    # YAWN DETECTION
                    # -------------------------

                    # Use full webcam frame
                    yawn_prob = predict_image(
                        yawn_model,
                        frame
                    )

                    if yawn_prob is not None:

                        no_yawn_prob = 1 - yawn_prob

                        if yawn_prob >= 0.5:
                            yawn_status = "YAWN"
                            yawn_color = (0, 0, 255)
                        else:
                            yawn_status = "NO YAWN"
                            yawn_color = (0, 255, 0)

                        cv2.putText(
                            frame,
                            f"Mouth: {yawn_status}",
                            (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            yawn_color,
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Yawn Probability: {yawn_prob:.3f}",
                            (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"No Yawn Probability: {no_yawn_prob:.3f}",
                            (20, 185),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 0),
                            2
                        )

            else:

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
                "Driver Drowsiness Detection V2",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("Camera closed successfully!")
