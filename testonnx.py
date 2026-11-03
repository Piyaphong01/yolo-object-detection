# ============================================================
# Script to run ONNX model on Raspberry Pi (or Windows for quick testing)
# Read video -> Detect objects -> Draw boxes -> Save video + CSV -> Print FPS summary
# ============================================================

import cv2                  # Image/video library (read video, draw boxes, write text, save video)
import numpy as np          # Array library (images are multi-dimensional arrays, model needs array input)
import onnxruntime as ort   # Library to load and run .onnx models, shortened to ort
import time                 # Timing library, used to calculate FPS
import csv                  # Standard library for writing CSV files (tables)

# ---------- CONFIG (edit these to match your actual files) ----------
MODEL_PATH = "best.onnx"          # ONNX model filename, must be in the same folder as this script
VIDEO_PATH = "test_video.mp4"     # Test video filename (use 0 instead to use a USB webcam)
OUTPUT_VIDEO_PATH = "result_output.mp4"  # Output video filename that will be created
OUTPUT_CSV_PATH = "detections.csv"       # CSV filename to store every detection box's coordinates
IMG_SIZE = 640                    # Image size (pixels) the model expects, must match export imgsz exactly
CONF_THRESHOLD = 0.4              # Minimum confidence to accept a detection (per competition rules)
CLASS_NAMES = ["Car", "truck"]    # Class names, ordered by index 0,1 to match dataset_custom.yaml
# ----------------------------------------------------------

# Load the ONNX model into memory, forcing CPU execution
# (Raspberry Pi has no dedicated GPU, so CPUExecutionProvider must be specified)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Get the input layer's name (each model may name its input differently)
# Needed to feed the image into the correct input slot when calling session.run()
input_name = session.get_inputs()[0].name


def preprocess(frame, img_size):
    # This function prepares one raw frame into the format the ONNX model expects
    # Input: frame = raw image from OpenCV, img_size = size the model expects (e.g. 640)

    img = cv2.resize(frame, (img_size, img_size))
    # Resize the image to img_size x img_size (e.g. 640x640) because the model only accepts a fixed size

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert color order from BGR (OpenCV's default) to RGB (the format used during training)
    # Without this, colors would be swapped and the model would detect incorrectly

    img = img.astype(np.float32) / 255.0
    # Convert data type from int (0-255) to float32, then divide by 255
    # to normalize pixel values into the 0.0 - 1.0 range the model was trained on

    img = np.transpose(img, (2, 0, 1))
    # Swap dimension order from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
    # because ONNX/PyTorch models need CHW order, not the HWC order OpenCV uses

    img = np.expand_dims(img, axis=0)
    # Add a "batch" dimension, turning shape (3,640,640) into (1,3,640,640)
    # because the model is designed to always receive data as a batch, even a single image needs batch=1

    return img
    # Return the fully prepared image ready to feed into the model


def postprocess(output, frame_shape, img_size, conf_thres):
    # This function converts the model's raw output into a usable list of detection boxes
    # output shape: (1, num_detections, 6) -> each row is [x1, y1, x2, y2, conf, class_id]

    detections = output[0]
    # Extract all detections (drop the batch dimension, leaving shape (300, 6))

    h, w = frame_shape[:2]
    # Get the height (h) and width (w) of the original frame (before resizing to 640x640)

    scale_x, scale_y = w / img_size, h / img_size
    # Calculate the scale factor to convert box coordinates from the 640x640 image
    # back to the original frame's real size, since the model detected on the resized image

    results = []
    # Empty list to hold only the detections that pass the confidence threshold

    for det in detections:
        # Loop through each of the 300 candidate detection rows

        x1, y1, x2, y2, conf, cls_id = det[:6]
        # Unpack the 6 values from this row into separate variables
        # x1,y1 = top-left corner, x2,y2 = bottom-right corner, conf = confidence, cls_id = class id

        if conf < conf_thres:
            continue
            # Skip this box entirely if confidence is below the threshold (0.4)

        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        # Scale the coordinates back to match the original frame size

        results.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_id)))
        # Add this passing detection (with scaled coordinates) to the results list
        # Coordinates are converted to int since pixel positions must be whole numbers

    return results
    # Return the full list of detections that passed the threshold


# Open the source video file to start reading frames
cap = cv2.VideoCapture(VIDEO_PATH)

# Get the source video's FPS; fall back to 30 if it can't be read
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Get the source video's width in pixels, converted to int
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Get the source video's height in pixels, converted to int
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the codec used to encode the output video ("mp4v" is the standard codec for .mp4 files)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Create the output video writer, matching filename, codec, FPS, and frame size to the source
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, src_fps, (frame_width, frame_height))

# Open the CSV file in write mode to store detection coordinates
# newline="" prevents extra blank lines when opening CSV files on Windows
csv_file = open(OUTPUT_CSV_PATH, mode="w", newline="")

# Create a CSV writer from the opened file
csv_writer = csv.writer(csv_file)

# Write the header row once at the start of the CSV file
csv_writer.writerow(["frame", "class_name", "confidence", "x1", "y1", "x2", "y2"])

# Counter for how many frames have been processed so far, starts at 0
frame_count = 0

# Accumulator for total processing time across all frames (seconds), used for average FPS at the end
total_time = 0.0

# Main loop: keep reading and processing frames as long as the video is open
while cap.isOpened():

    ret, frame = cap.read()
    # Read one frame from the video
    # ret = True/False whether the read succeeded, frame = the image data for that frame (None if failed)

    if not ret:
        break
        # If reading failed (video ended, or file not found), exit the loop immediately

    t0 = time.time()
    # Record the current time (timestamp) as the starting point for this frame's timing

    img_input = preprocess(frame, IMG_SIZE)
    # Call preprocess() to prepare this frame for the model

    outputs = session.run(None, {input_name: img_input})
    # Run inference on the model
    # Feed the prepared image in via input_name, and get the raw output back into outputs

    detections = postprocess(outputs[0], frame.shape, IMG_SIZE, CONF_THRESHOLD)
    # Call postprocess() to turn the raw output into a usable list of detection boxes

    t1 = time.time()
    # Record the current time again (the end point for this frame's processing)

    frame_time = t1 - t0
    # Calculate how long this frame took to process in total (preprocess + inference + postprocess)

    total_time += frame_time
    # Add this frame's time to the running total

    frame_count += 1
    # Increment the frame counter by 1

    fps = 1.0 / frame_time if frame_time > 0 else 0
    # Calculate the FPS for just this single frame (1 divided by the time it took)
    # Guard against division by zero in case the measured time is exactly 0

    # Loop through every object detected in this frame: draw box + write coordinates to CSV
    for (x1, y1, x2, y2, conf, cls_id) in detections:

        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        # Build the label text, e.g. "Car 0.93" (class name + confidence to 2 decimal places)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw a green rectangle (BGR: 0,255,0) around the detected object, 2 pixel line thickness

        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Write the label text slightly above the box (y1-10), in green, at font scale 0.6

        csv_writer.writerow([frame_count, CLASS_NAMES[cls_id], f"{conf:.4f}", x1, y1, x2, y2])
        # Write one row for this detection to the CSV: frame number, class name, confidence, x1,y1,x2,y2

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Write this frame's FPS value in the top-left corner of the image, in red, at font scale 1

    out.write(frame)
    # Save this frame (with boxes, labels, and FPS drawn on it) to the output video

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")
        # Every 30 frames (evenly divisible by 30), print a progress update to the terminal

# ---- End of main loop (video finished, or read failed) ----

cap.release()
# Close the connection to the source video file, freeing system resources

out.release()
# Finalize and close the output video file (important - if not closed, the file may be corrupted)

csv_file.close()
# Close the CSV file (also important, to make sure all data is fully written to disk)

avg_fps = frame_count / total_time if total_time > 0 else 0
# Calculate the average FPS across the whole video (total frames divided by total time)

print(f"\n=== Summary ===")
# Print the summary header, with a blank line before it for readability

print(f"Total frames: {frame_count}")
# Show the total number of frames that were processed

if frame_count > 0:
    # Check that at least one frame was processed (avoids division by zero if the video failed to read)

    print(f"Average time per frame: {total_time / frame_count * 1000:.2f} ms")
    # Show the average time per frame, converted from seconds to milliseconds (multiply by 1000)

    print(f"Average FPS: {avg_fps:.2f}")
    # Show the average FPS calculated above, to 2 decimal places

print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
# Show the filename/location of the saved output video

print(f"Detection coordinates saved to: {OUTPUT_CSV_PATH}")
# Show the filename/location of the saved CSV file with all detection coordinates