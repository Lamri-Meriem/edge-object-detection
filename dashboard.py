from ultralytics import YOLO
import csv
import time

# Load the quantized model (fast + small)
model = YOLO("C:/Users/LENOVO/yolov8n_int8.onnx")

video_path = "C:/Users/LENOVO/Downloads/tst.mp4"
output_csv = "C:/Users/LENOVO/Downloads/detection_log.csv"

# Run detection on every frame, stream=True processes frame-by-frame instead of all at once
results = model(video_path, stream=True)

log_rows = []
frame_number = 0
start = time.time()

for r in results:
    frame_number += 1
    # Count how many of each object type appeared in this frame
    counts = {}
    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        counts[class_name] = counts.get(class_name, 0) + 1

    total_objects = sum(counts.values())
    row = {
        "frame": frame_number,
        "total_objects": total_objects,
        "details": counts
    }
    log_rows.append(row)
    print(f"Frame {frame_number}: {total_objects} objects — {counts}")

end = time.time()

# Save everything to a CSV file
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "total_objects", "details"])
    for row in log_rows:
        writer.writerow([row["frame"], row["total_objects"], row["details"]])

print(f"\nDone. Processed {frame_number} frames in {end - start:.1f} seconds.")
print(f"Log saved to: {output_csv}")