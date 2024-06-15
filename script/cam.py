#!/usr/bin/env python3

import numpy as np
import cv2
import torch
import os
import time
import argparse
from model import create_model
from config import NUM_CLASSES, DEVICE, CLASSES

# Environment ROS
import rospy
from sensor_msgs.msg import CameraInfo
camera = 2

np.random.seed(42)

# Construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--imgsz', default=300, type=int, help='image resize shape')
parser.add_argument('--threshold', default=0.7, type=float, help='detection threshold')
args = vars(parser.parse_args())

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def load_model(checkpoint_path):
    """
    Load the trained model.
    """
    model = create_model(num_classes=NUM_CLASSES, size=300)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    return model

def preprocess_frame(frame, img_size):
    """
    Preprocess the frame for model prediction.
    """
    image = frame.copy()
    if img_size is not None:
        image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image_input = torch.tensor(image_input, dtype=torch.float).unsqueeze(0)
    return image_input

def rescale_box_coordinates(box, original_size, resized_size):
    """
    Rescale the box coordinates based on the resized and original size.
    """
    xmin = int((box[0] / resized_size[0]) * original_size[0])
    ymin = int((box[1] / resized_size[1]) * original_size[1])
    xmax = int((box[2] / resized_size[0]) * original_size[0])
    ymax = int((box[3] / resized_size[1]) * original_size[1])
    return xmin, ymin, xmax, ymax

def draw_bounding_boxes(frame, boxes, pred_classes, scores, colors, threshold, original_size, resized_size):
    """
    Draw bounding boxes and class labels on the frame.
    """
    midpoint_drawn = False
    frame_center_x, frame_center_y = original_size[0] // 2, original_size[1] // 2

    for j, box in enumerate(boxes):
        if scores[j] < threshold:
            continue
        class_name = pred_classes[j]
        score = scores[j]
        display_name = f"{class_name} {score:.2f}"
        color = colors[CLASSES.index(class_name)]
        
        # Rescale the box coordinates
        xmin, ymin, xmax, ymax = rescale_box_coordinates(box, original_size, resized_size)
        
        center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
        
        if not midpoint_drawn:
            cv2.line(frame, (center_x, center_y), (frame_center_x, frame_center_y), (255, 0, 0), 2)
            midpoint_drawn = True

        cv2.circle(frame, (frame_center_x, frame_center_y), 7, (255, 0, 0), -1)
        cv2.circle(frame, (center_x, center_y), 7, (255, 255, 255), -1)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color[::-1], 2)
        cv2.putText(frame, display_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color[::-1], 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, str((center_x, center_y)), (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, color[::-1], 2, lineType=cv2.LINE_AA)

def extract_and_show_detected_objects(frame, boxes, scores, pred_classes, threshold, original_size, resized_size, open_windows):
    """
    Extract and show detected objects in separate windows. Close windows if no object is detected.
    """
    current_open_windows = set()
    for j, box in enumerate(boxes):
        if scores[j] < threshold:
            continue
        
        class_name = pred_classes[j]
        
        # Rescale the box coordinates
        xmin, ymin, xmax, ymax = rescale_box_coordinates(box, original_size, resized_size)
        
        detected_object = frame[ymin:ymax, xmin:xmax]
        current_open_windows.add(f"Object: {class_name}")
    
    # Close windows that are no longer needed
    for window in open_windows - current_open_windows:
        cv2.destroyWindow(window)
    
    return current_open_windows

def process_video(model, img_size, threshold):
    """
    Process the video frame by frame and perform object detection.
    """
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print('Error while trying to read video. Please check the path again')
        return

    frame_count, total_fps = 0, 0
    open_windows = set()
    last_print_time = time.time()

    # Initialize ROS publisher
    pub = rospy.Publisher("/camera_info", CameraInfo, queue_size=10)
    rospy.init_node("camera_info_publisher", anonymous=True)
    rate = rospy.Rate(5)  # 5 Hz

    while not rospy.is_shutdown() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure the frame is not mirrored
        frame = cv2.flip(frame, 1)  # Flip frame horizontally to correct mirroring

        original_size = frame.shape[1], frame.shape[0]  # width, height
        image_input = preprocess_frame(frame, img_size)
        resized_size = img_size, img_size  # New resized dimensions

        start_time = time.time()
        with torch.no_grad():
            outputs = model(image_input.to(DEVICE))
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        if outputs and len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].numpy()
            scores = outputs[0]['scores'].numpy()
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].numpy()]
            
            # Check if it's time to print and if the score meets the threshold
            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                for i, score in enumerate(scores):
                    if score >= threshold:
                        print(f"First predicted class: {pred_classes[i]}, Score: {score:.2f}")
                        last_print_time = current_time
                        break

            draw_bounding_boxes(frame, boxes, pred_classes, scores, COLORS, threshold, original_size, resized_size)
            open_windows = extract_and_show_detected_objects(frame, boxes, scores, pred_classes, threshold, original_size, resized_size, open_windows)

            # Publish the center of the first detected object as CameraInfo
            center_x, center_y = None, None
            for j, box in enumerate(boxes):
                if scores[j] >= threshold:
                    xmin, ymin, xmax, ymax = rescale_box_coordinates(box, original_size, resized_size)
                    center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
                    break

            if center_x is not None and center_y is not None:
                msg = CameraInfo()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "camera_frame"
                msg.height = original_size[1]
                msg.width = original_size[0]
                msg.K[2] = center_x
                msg.K[5] = center_y
                pub.publish(msg)
                rospy.loginfo(f"Published center: x={center_x}, y={center_y}")
        else:
            # Close all windows if no objects are detected
            for window in open_windows:
                cv2.destroyWindow(window)
            open_windows = set()

        cv2.putText(frame, f"{fps:.0f} FPS", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Video Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()

def main():
    checkpoint_path = 'script/outputs/best_model.pth'
    model = load_model(checkpoint_path)
    process_video(model, args['imgsz'], args['threshold'])

if __name__ == "__main__":
    main()
