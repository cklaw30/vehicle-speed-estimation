# Traffic Monitoring and Vehicle Speed Estimation with Application 

## Overview
This is a **Traffic Monitoring and Vehicle Speed Estimation with Application** project using **YOLOv8**, **OpenCV**, and **Streamlit**. It processes uploaded videos, detects vehicles, estimates their speeds, and provides a dashboard for visualization.

## Features
- Upload video files for processing.
- Detect and track vehicles using **YOLOv8**.
- Estimate vehicle speeds.
- View real-time metrics (current count, total vehicles, and average speed).
- Download processed results, including an annotated video and a JSON log.
- User-friendly web interface powered by **Streamlit**.

## Installation
### Dataset
download the dataset from here: https://mmuedumy-my.sharepoint.com/:f:/g/personal/1211103427_student_mmu_edu_my/EvmXO1DUVplDkvaj0IGVwh8BlQxkXs2RL15aFbBifpo1Pg?e=rmOoLr

### Prerequisites
Ensure you have Python 3.8 or later installed.

### Install Dependencies
Run the following command to install all required packages:

```bash
pip install streamlit opencv-python numpy ultralytics supervision torch pandas
```

### YOLOv8 Model Setup
The application uses YOLOv8 for object detection. Ensure the model file (`yolov8x.pt`) is downloaded automatically when first used or manually download it from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).

## Running the Application
Run the Streamlit application with the following command:

```bash
streamlit run app.py
```

## Usage
1. **Upload a Video File**: Upload an `.mp4`, `.avi`, or `.mov` file in the Streamlit sidebar.
2. **Upload a Calibration File**: Provide a JSON file containing transformation parameters.
3. **Start Processing**: Click the "Start Processing" button to begin vehicle detection.
4. **Monitor the Dashboard**:
   - View real-time metrics (current vehicles, total vehicles, average speed).
   - Watch the processed video with bounding boxes and speed annotations.
   - Examine the vehicle log.
5. **Download Results**: Download a ZIP file containing the processed video and JSON log.

## Calibration File Format
The calibration JSON file must have the following structure:

```json
{
    "SOURCE": [
        [608, 246],
        [1050, 246],
        [1610, 1077],
        [76, 1077]
    ],
    "TARGET_WIDTH": 10,
    "TARGET_HEIGHT": 73
}
```

- `SOURCE`: 4 corner points defining the region for perspective transformation.
- `TARGET_WIDTH` & `TARGET_HEIGHT`: Dimensions of the transformed view.

## Acknowledgments
- **Ultralytics YOLO** for object detection.
- **OpenCV** for image processing.
- **Streamlit** for interactive UI.
- **Supervision** library for tracking and annotations.