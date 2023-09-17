from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Function to load class names from obj.names file
def load_class_names(names_file):
    with open(names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Load YOLO models
model_pose = YOLO('/media/ali/Files/models/yolov8/pose_estimation/yoga_82/ultralytics/runs/pose/syoga_82_latest/weights/best.pt')

# Input and output video paths
input_video_path = 'test.mp4'
output_video_path = 'output_video.mp4'

# Open the input video file
input_video = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(input_video.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Load class names from obj.names file
class_names = load_class_names('obj.names')

# Define the connections between COCO keypoints
connections = [
    (1, 2),  # Connect left eye to right eye
    (1, 5),  # Connect left eye to left shoulder
    (2, 6),  # Connect right eye to right shoulder
    (5, 7),  # Connect left shoulder to left elbow
    (7, 9),  # Connect left elbow to left wrist
    (6, 8),  # Connect right shoulder to right elbow
    (8, 10),  # Connect right elbow to right wrist
    (5, 11),  # Connect left shoulder to left hip
    (6, 12),  # Connect right shoulder to right hip
    (11, 12),  # Connect left hip to right hip
    (11, 13),  # Connect left hip to left knee
    (13, 15),  # Connect left knee to left ankle
    (12, 14),  # Connect right hip to right knee
    (14, 16),  # Connect right knee to right ankle
    (0, 1),  # Connect nose to left eye
    (0, 2),  # Connect nose to right eye
    (0, 3),  # Connect nose to left ear
    (0, 4)  # Connect nose to right ear
]

# Define colors for keypoints and connections
keypoint_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255),
                   (0, 128, 255), (0, 255, 128), (128, 0, 255), (128, 255, 0), (255, 0, 128), (255, 128, 0),
                   (0, 64, 128), (0, 128, 64), (64, 0, 128), (64, 128, 0), (128, 0, 64), (128, 64, 0)]

connection_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255),
                     (0, 128, 255), (0, 255, 128), (128, 0, 255), (128, 255, 0), (255, 0, 128), (255, 128, 0),
                     (0, 64, 128), (0, 128, 64), (64, 0, 128), (64, 128, 0), (128, 0, 64)]

# Process each frame of the input video
while input_video.isOpened():
    # Read a frame from the video
    ret, frame = input_video.read()

    if not ret:
        break
    
    res_plotted = None
    # Perform inference with YOLO pose model
    results = model_pose(frame)
    
    keypoints = results[0].keypoints
    boxes = results[0].boxes
    print(boxes.conf)
   
    # Get keypoints from the results
    keypoints_data = keypoints.data  # raw keypoints tensor, (num_dets, num_kpts, 2/3)
    if len(boxes) > 0 and max(boxes.conf) >= 0.90:
        max_conf_index = torch.argmax(boxes.conf.cpu()).item()
        max_conf_class = class_names[max_conf_index]

        if keypoints_data.size(1) > 0:
            res_plotted = results[0].plot()
            # Plot connected keypoints on the frame
            for i in range(keypoints_data.shape[0]):
                for j in range(keypoints_data.shape[1]):
                    x, y = keypoints_data[i, j, 0], keypoints_data[i, j, 1]
                    color = keypoint_colors[j % len(keypoint_colors)]  # Select color from the keypoint color list
                    cv2.circle(res_plotted, (int(x), int(y)), 3, color, -1)  # Draw a circle for each keypoint

                for connection_idx, connection in enumerate(connections):
                    start_idx, end_idx = connection
                    if start_idx < keypoints_data.shape[1] and end_idx < keypoints_data.shape[1]:
                        start_x, start_y = keypoints_data[i, start_idx, 0], keypoints_data[i, start_idx, 1]
                        end_x, end_y = keypoints_data[i, end_idx, 0], keypoints_data[i, end_idx, 1]
                        color = connection_colors[connection_idx % len(connection_colors)]  # Select color from the connection color list
                        cv2.line(res_plotted, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color, 2)  # Connect keypoints with a line



    if res_plotted is not None:
        # Write the processed frame to the output video
        output_video.write(res_plotted)
   

# Release the video capture and writer objects
input_video.release()
output_video.release()
