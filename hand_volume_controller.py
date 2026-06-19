import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Volume control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin, volMax = volume.GetVolumeRange()[:2]

# MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            lmList = []
            h, w, c = img.shape

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            if len(lmList) > 8:

                x1, y1 = lmList[4]   # Thumb tip
                x2, y2 = lmList[8]   # Index tip

                cv2.circle(img, (x1, y1), 10, (255, 0, 255), -1)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), -1)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                length = math.hypot(x2 - x1, y2 - y1)

                # Convert finger distance to volume
                vol = volMin + (length / 200) * (volMax - volMin)

                if vol < volMin:
                    vol = volMin

                if vol > volMax:
                    vol = volMax

                volume.SetMasterVolumeLevel(vol, None)

                volPercent = int((length / 200) * 100)

                if volPercent < 0:
                    volPercent = 0

                if volPercent > 100:
                    volPercent = 100

                cv2.putText(
                    img,
                    f"Volume: {volPercent}%",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Gesture Volume Controller", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()