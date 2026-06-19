import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_points, w, h):
    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]

    p1 = (int(p1.x*w), int(p1.y*h))
    p2 = (int(p2.x*w), int(p2.y*h))
    p3 = (int(p3.x*w), int(p3.y*h))
    p4 = (int(p4.x*w), int(p4.y*h))
    p5 = (int(p5.x*w), int(p5.y*h))
    p6 = (int(p6.x*w), int(p6.y*h))

    vertical1 = math.dist(p2, p5)
    vertical2 = math.dist(p3, p6)
    horizontal = math.dist(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

cap = cv2.VideoCapture(0)

closed_frames = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:

        for face in results.multi_face_landmarks:

            landmarks = face.landmark

            left_ear = eye_aspect_ratio(
                landmarks,
                LEFT_EYE,
                w,
                h
            )

            right_ear = eye_aspect_ratio(
                landmarks,
                RIGHT_EYE,
                w,
                h
            )

            ear = (left_ear + right_ear) / 2

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            if ear < 0.22:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames > 15:
                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()