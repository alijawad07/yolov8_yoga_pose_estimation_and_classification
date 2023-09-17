# YOLOv8 Yoga Pose Estimation and Classification

This repository contains a YOLOv8-based solution for detecting and classifying various yoga poses using the Yoga82 dataset.

## Project Overview:
The project utilizes the YOLOv8 architecture to achieve pose estimation and yoga posture classification in real-time. The initial training data is derived from the Yoga82 dataset which was further processed to fit the needs of pose estimation.

## Features:
- **Pose Estimation**: Detect human poses in real-time from video streams.
- **Yoga Pose Classification**: Classify the detected poses into specific yoga postures.
- **Visualization**: Highlight key points and their connections for easier understanding of postures.

## How to Use:
1. Ensure you have all the required libraries installed. This includes `ultralytics`, `cv2`, `numpy`, and `torch`.
2. Load your trained YOLOv8 model using the given path.
3. Specify the input and output video paths.
4. Run the provided script to process the video and obtain an output with the estimated poses and classifications.

## Dependencies:
- OpenCV
- PyTorch
- Ultralytics YOLO

## Dataset:
The project is trained on a modified version of the Yoga82 dataset, which has been processed for compatibility with YOLOv8 pose estimation.

## Future Work:
- Improve accuracy by integrating additional datasets.
- Integrate real-time feedback for yoga posture correction.

## License:
This project is open-source and available under the MIT License.

## Acknowledgements:
- Special thanks to the creators of the Yoga82 dataset.
- The Ultralytics team for the YOLO framework.

Contributions are welcome! Feel free to raise issues or submit pull requests.
