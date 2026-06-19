from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

# Load video
cap = cv2.VideoCapture("helmet.mp4")

if not cap.isOpened():
    print("ERROR: Could not open helmet.mp4")
    exit()

helmet_count = 0
no_helmet_count = 0

print("Video started... Press ESC to stop.\n")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()

            # Count detections
            if label in ["helmet", "with helmet"]:
                helmet_count += 1

            elif label in ["no_helmet", "without helmet", "no helmet"]:
                no_helmet_count += 1

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("Helmet Detection", annotated_frame)

    # Press ESC to stop
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\n==============================")
print(" HELMET DETECTION REPORT")
print("==============================")
print("Helmet Detected    :", helmet_count)
print("No Helmet Detected :", no_helmet_count)
print("Total Detections   :", helmet_count + no_helmet_count)
print("==============================")