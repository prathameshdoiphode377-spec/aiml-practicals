# Practical 9: Object Detection from CCTV Footage

## Aim
To apply object detection to CCTV footage using a pretrained YOLOv8 model, detecting and counting objects such as people and vehicles in every frame.

## Model and input
- **Model:** YOLOv8 nano (`yolov8n.pt`), pretrained on the COCO dataset (80 classes including person, car, bus, truck, motorcycle, bicycle).
- **Input:** a short CCTV-style video, `cctv_sample.mp4` (not uploaded to this repo because of file size).

## Theory
- **Object detection** finds *what* objects are in an image and *where* they are. The output is a **bounding box**, a **class label** and a **confidence score** for each object. It is different from image classification (one label per image) and segmentation (a label per pixel).
- **YOLO (You Only Look Once)** is a single-stage detector. It looks at the whole image once and predicts boxes and classes together, which makes it fast enough for real-time video. Two-stage detectors like Faster R-CNN first propose regions and then classify them, which is usually slower.
- **How YOLO works:**
  1. The image is resized and passed through a convolutional neural network (CNN).
  2. The network predicts many bounding boxes with class probabilities.
  3. Boxes below the **confidence threshold** are discarded.
  4. **Non-Maximum Suppression (NMS)** removes duplicate boxes for the same object, keeping the best one.
- **IoU (Intersection over Union)** = overlap area / union area of two boxes. It is used in NMS and to evaluate detections.
- **Pretrained model / transfer learning:** the model was already trained on the large COCO dataset, so it can be used directly without training from scratch.
- **Video = frames:** a video is processed as a sequence of images (frames), and detection is run on each frame.
- **Tracking (optional):** assigns a persistent ID to each object across frames, so unique people and vehicles can be counted instead of counting the same person in every frame.
- **Metrics used to evaluate detectors:** precision, recall and mAP (mean average precision).
- **Applications of CCTV detection:** people counting, intrusion detection, traffic monitoring, parking management, crowd analysis.

## Steps
1. Install the libraries and get a short video.
2. Load the pretrained YOLOv8 model.
3. Open the video with OpenCV and read it frame by frame.
4. Run the detector on each frame with a confidence threshold of 0.4.
5. Draw boxes, labels and confidence scores on each frame and write them to an output video.
6. Count persons and vehicles per frame, and save sample frames.
7. Plot the counts, print the summary and state the conclusion.

## How to run
```bash
pip install -r requirements.txt
python object_detection_cctv.py --video cctv_sample.mp4 | Tee-Object output.txt
```
Optional: add `--track` to also count unique objects by ID.

The first run downloads `yolov8n.pt` (about 6 MB) automatically, so an internet connection is needed once.

## Output
Run the script, then the files below appear in the same folder.

### Sample detected frames
![Sample frame 1](sample_frame_1.jpg)
![Sample frame 2](sample_frame_2.jpg)
![Sample frame 3](sample_frame_3.jpg)

### Objects detected per frame
![Detection counts](detection_counts.png)

### Results
(Fill these in from your `output.txt` after running.)

| Measure | Value |
|---|---|
| Video length / frames | |
| Frames processed per second | |
| Average persons per frame | |
| Maximum persons in a frame | |
| Average vehicles per frame | |
| Most detected class | |

The full console output is in [output.txt](output.txt).

## Conclusion
(Write 2 to 3 lines using your own numbers, for example: YOLOv8 detected people and vehicles in the CCTV video frame by frame, with an average of __ persons and __ vehicles per frame. It ran at __ frames per second on the CPU. Detection can be affected by low light, small or distant objects and occlusion.)
