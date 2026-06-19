import cv2

cap = cv2.VideoCapture("helmet.mp4")  # or your actual filename

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video not opened or video ended")
        break

    cv2.imshow("Video Test", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()