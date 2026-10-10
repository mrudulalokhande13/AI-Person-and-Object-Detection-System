import cv2
import csv
import os
import time
import threading
import pyttsx3

from datetime import datetime
from collections import Counter
from ultralytics import YOLO


# ==========================================
# 1. PROJECT SETUP
# ==========================================

print("Loading AI Object Detection Model...")

model = YOLO("yolo11n.pt")

# Create folders for screenshots and history
os.makedirs("screenshots", exist_ok=True)

history_file = "detection_history.csv"

# Create CSV file if it does not exist
if not os.path.exists(history_file):
    with open(history_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Date",
            "Time",
            "Object Name",
            "Count"
        ])


# ==========================================
# 2. VOICE ALERT SETUP
# ==========================================

voice_queue = []
voice_lock = threading.Lock()


def voice_worker():
    try:
        engine = pyttsx3.init()

        while True:
            with voice_lock:
                message = voice_queue.pop(0) if voice_queue else None

            if message:
                engine.say(message)
                engine.runAndWait()
            else:
                time.sleep(0.1)

    except Exception as error:
        print("Voice alert error:", error)


threading.Thread(
    target=voice_worker,
    daemon=True
).start()


def announce(message):
    with voice_lock:
        # Keep the queue small so old alerts don't pile up
        if len(voice_queue) < 2:
            voice_queue.append(message)


# ==========================================
# 3. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    raise SystemExit


print("AI Person and Object Detection Started!")
print("S = Save Screenshot")
print("Q = Quit")


# ==========================================
# 4. MAIN DETECTION LOOP
# ==========================================

previous_person_count = 0
previous_objects = set()

last_history_time = 0
history_interval = 5

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to read webcam.")
        break

    # Detect objects using YOLO
    results = model(
        frame,
        conf=0.5,
        verbose=False
    )

    result = results[0]

    # Draw boxes and labels
    output = result.plot()

    # Count objects in the current frame
    counts = Counter()

    for box in result.boxes:
        class_id = int(box.cls[0])
        object_name = model.names[class_id]
        counts[object_name] += 1

    # ======================================
    # FEATURE 1: PERSON COUNTING
    # ======================================

    person_count = counts.get("person", 0)

    cv2.putText(
        output,
        f"People Count: {person_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ======================================
    # FEATURE 2: OBJECT COUNTING
    # ======================================

    y_position = 70

    for name, count in sorted(counts.items()):

        if name == "person":
            continue

        cv2.putText(
            output,
            f"{name}: {count}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        y_position += 25

    # ======================================
    # FEATURE 3: SCREENSHOT CAPTURE
    # ======================================

    cv2.putText(
        output,
        "S: Screenshot | Q: Quit",
        (20, output.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ======================================
    # FEATURE 4: VOICE ALERTS
    # ======================================

    # Announce when a person first appears
    if person_count > 0 and previous_person_count == 0:
        announce("Person detected")

    # Announce when a new object category appears
    current_objects = set(counts.keys())

    new_objects = current_objects - previous_objects

    for name in new_objects:
        if name != "person":
            announce(name + " detected")

    previous_person_count = person_count
    previous_objects = current_objects

    # ======================================
    # FEATURE 5: DETECTION HISTORY
    # ======================================

    current_time = time.time()

    # Save detection counts every 5 seconds
    if current_time - last_history_time >= history_interval:

        now = datetime.now()

        with open(
            history_file,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            if counts:
                for name, count in sorted(counts.items()):
                    writer.writerow([
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        name,
                        count
                    ])
            else:
                writer.writerow([
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    "No objects",
                    0
                ])

        last_history_time = current_time

    # ======================================
    # DISPLAY RESULT
    # ======================================

    cv2.imshow(
        "AI Person and Object Detection",
        output
    )

    key = cv2.waitKey(1) & 0xFF

    # Save screenshot when S is pressed
    if key == ord("s"):

        filename = datetime.now().strftime(
            "screenshots/detection_%Y%m%d_%H%M%S.jpg"
        )

        cv2.imwrite(filename, output)

        print("Screenshot saved:", filename)

    # Exit when Q is pressed
    elif key == ord("q"):
        break


# ==========================================
# CLOSE PROJECT
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Project stopped.")
print("Detection history:", history_file)
print("Screenshots folder: screenshots")
