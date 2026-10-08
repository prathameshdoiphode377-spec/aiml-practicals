"""
Practical 9: Apply object detection from CCTV footage
Model   : YOLOv8 (pretrained on the COCO dataset, 80 object classes)
Input   : a short CCTV-style video (traffic, street, shop, corridor ...)
Output  : annotated video, 3 sample frames, a detection-count graph and a summary

Usage   : python object_detection_cctv.py --video cctv_sample.mp4
Options : --model yolov8n.pt   (n = nano/fast, s = small, m = medium)
          --conf 0.4           (minimum confidence to keep a detection)
          --track              (give each object an ID to count UNIQUE objects)
"""

import argparse
import csv
import time
from collections import Counter, defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")           # remove this line if you run in Jupyter / Colab
import matplotlib.pyplot as plt
from ultralytics import YOLO

PERSON = "person"
VEHICLES = {"car", "motorcycle", "bus", "truck", "bicycle"}

# ---------------------------------------------------------------
# Step 1: Read the command-line options
# ---------------------------------------------------------------
parser = argparse.ArgumentParser(description="Object detection on CCTV footage with YOLOv8")
parser.add_argument("--video", default="cctv_sample.mp4", help="path to the input video")
parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights file")
parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
parser.add_argument("--track", action="store_true", help="track objects to count unique ones")
args = parser.parse_args()

# ---------------------------------------------------------------
# Step 2: Load the pretrained model (downloads the weights the first time)
# ---------------------------------------------------------------
model = YOLO(args.model)
print("Model loaded:", args.model)
print("Number of classes the model can detect:", len(model.names))

# ---------------------------------------------------------------
# Step 3: Open the input video and create the output video
# ---------------------------------------------------------------
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f"Could not open video '{args.video}'. Check the file name and path.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {args.video} | {width}x{height} | {fps:.1f} FPS | {total_frames} frames")

writer = cv2.VideoWriter("output_detected.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# ---------------------------------------------------------------
# Step 4: Detect objects frame by frame
# ---------------------------------------------------------------
class_counts = Counter()            # total detections per class (summed over all frames)
unique_ids = defaultdict(set)       # unique tracked IDs per class (only with --track)
persons_per_frame, vehicles_per_frame = [], []
sample_positions = {total_frames // 4, total_frames // 2, (3 * total_frames) // 4}
saved_samples = 0

frame_no = 0
start = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_no += 1

    if args.track:
        result = model.track(frame, persist=True, conf=args.conf, verbose=False)[0]
    else:
        result = model(frame, conf=args.conf, verbose=False)[0]

    n_person, n_vehicle = 0, 0
    for box in result.boxes:
        name = model.names[int(box.cls)]
        class_counts[name] += 1
        if name == PERSON:
            n_person += 1
        elif name in VEHICLES:
            n_vehicle += 1
        if args.track and box.id is not None:
            unique_ids[name].add(int(box.id))
    persons_per_frame.append(n_person)
    vehicles_per_frame.append(n_vehicle)

    annotated = result.plot()                   # draws boxes, labels and confidence scores
    writer.write(annotated)

    if frame_no in sample_positions and saved_samples < 3:
        saved_samples += 1
        cv2.imwrite(f"sample_frame_{saved_samples}.jpg", annotated)

    if frame_no % 50 == 0:
        print(f"Processed {frame_no}/{total_frames} frames")

cap.release()
writer.release()
elapsed = time.time() - start

# ---------------------------------------------------------------
# Step 5: Summary of the results
# ---------------------------------------------------------------
print("\n" + "=" * 55)
print("DETECTION SUMMARY")
print("=" * 55)
print(f"Frames processed          : {frame_no}")
print(f"Processing time           : {elapsed:.1f} s  ({frame_no / elapsed:.1f} frames/s)")
print(f"Average persons per frame : {sum(persons_per_frame) / max(frame_no, 1):.2f}")
print(f"Maximum persons in a frame: {max(persons_per_frame, default=0)}")
print(f"Average vehicles per frame: {sum(vehicles_per_frame) / max(frame_no, 1):.2f}")
print(f"Maximum vehicles in a frame: {max(vehicles_per_frame, default=0)}")

print("\nTotal detections per class (summed over all frames):")
for name, count in class_counts.most_common(10):
    print(f"  {name:<15}{count}")

if args.track:
    print("\nUnique objects (by tracking ID):")
    for name, ids in sorted(unique_ids.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"  {name:<15}{len(ids)}")

# Save per-frame counts as a CSV file
with open("detection_counts.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "persons", "vehicles"])
    for i, (p, v) in enumerate(zip(persons_per_frame, vehicles_per_frame), start=1):
        w.writerow([i, p, v])

# ---------------------------------------------------------------
# Step 6: Graph of objects per frame
# ---------------------------------------------------------------
plt.figure(figsize=(10, 4.5))
plt.plot(persons_per_frame, label="Persons")
plt.plot(vehicles_per_frame, label="Vehicles")
plt.title("Objects detected per frame")
plt.xlabel("Frame number"); plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("detection_counts.png", dpi=150)
plt.close()

print("\nFiles created: output_detected.mp4, sample_frame_1-3.jpg, detection_counts.png, detection_counts.csv")
print("\nConclusion: YOLOv8 detected and labelled objects in the CCTV video frame by frame, "
      "and the counts of persons and vehicles can be used for monitoring and traffic analysis.")
