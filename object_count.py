from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("traffic.mp4")

if not cap.isOpened():
    print("ERROR: Could not open traffic.mp4")
    exit()

# Object counters
person_count = 0
car_count = 0
motorcycle_count = 0
bus_count = 0
truck_count = 0

print("Video started... Press ESC to stop.\n")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    frame_person = 0
    frame_car = 0
    frame_motorcycle = 0
    frame_bus = 0
    frame_truck = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                frame_person += 1

            elif label == "car":
                frame_car += 1

            elif label == "motorcycle":
                frame_motorcycle += 1

            elif label == "bus":
                frame_bus += 1

            elif label == "truck":
                frame_truck += 1

    # Store highest count seen in video
    person_count = max(person_count, frame_person)
    car_count = max(car_count, frame_car)
    motorcycle_count = max(motorcycle_count, frame_motorcycle)
    bus_count = max(bus_count, frame_bus)
    truck_count = max(truck_count, frame_truck)

    annotated_frame = results[0].plot()

    total = (
        frame_person
        + frame_car
        + frame_motorcycle
        + frame_bus
        + frame_truck
    )

    cv2.putText(
        annotated_frame,
        f"Total Objects: {total}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Object Detection & Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\n==============================")
print(" OBJECT DETECTION REPORT")
print("==============================")
print("Persons     :", person_count)
print("Cars        :", car_count)
print("Motorcycles :", motorcycle_count)
print("Buses       :", bus_count)
print("Trucks      :", truck_count)
print("==============================")
print(
    "Total Objects :",
    person_count
    + car_count
    + motorcycle_count
    + bus_count
    + truck_count,
)
print("==============================")