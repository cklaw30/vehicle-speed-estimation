import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import warnings
import torch
from collections import defaultdict, deque
import json
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime
import time
import os
import zipfile

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None

# Define vehicle classes from COCO dataset
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

class VehicleTracker:
    def __init__(self):
        self.vehicle_log = {}  # Store complete vehicle info
        self.current_count = 0  # Current vehicles in frame
        self.total_vehicles = 0  # Total unique vehicles detected
        
    def update_vehicle(self, vehicle_id, speed, frame_time, vehicle_type, has_enough_data):
        # Only update vehicle log if we have enough data points for speed calculation
        if has_enough_data:
            if vehicle_id not in self.vehicle_log:
                # New vehicle
                self.vehicle_log[vehicle_id] = {
                    'vehicle_type': vehicle_type,
                    'first_seen': frame_time,
                    'speeds': [speed] if speed else [],
                    'avg_speed': speed if speed else 0,
                    'last_updated': frame_time
                }
                self.total_vehicles += 1
            else:
                # Update existing vehicle
                if speed:
                    self.vehicle_log[vehicle_id]['speeds'].append(speed)
                    self.vehicle_log[vehicle_id]['avg_speed'] = np.mean(self.vehicle_log[vehicle_id]['speeds'])
                self.vehicle_log[vehicle_id]['last_updated'] = frame_time
    
    def get_vehicle_data(self):
        return pd.DataFrame([
            {
                'Vehicle ID': vid,
                'Vehicle Type': data['vehicle_type'],
                'First Seen': data['first_seen'].strftime('%H:%M:%S'),
                'Average Speed (km/h)': f"{data['avg_speed']:.1f}",
                'Last Updated': data['last_updated'].strftime('%H:%M:%S')
            }
            for vid, data in self.vehicle_log.items()
        ])

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.info(f"Using device: {device}")
    
    model = YOLO("yolov8x.pt")
    model.to(device)
    return model

def create_dashboard():
    # Create metrics at the top
    st.markdown("### Real-time Metrics")
    metric_cols = st.columns(3)
    current_count = metric_cols[0].empty()
    total_vehicles = metric_cols[1].empty()
    avg_speed = metric_cols[2].empty()
    
    # Add some spacing
    st.markdown("---")
    
    # Create two columns for video and log
    left_col, right_col = st.columns([2, 1])  # 2:1 ratio for video:log
    
    with left_col:
        # Create placeholder for video
        video_frame = st.empty()
        # Create placeholder for the video control buttons
        video_controls = st.empty()
    
    with right_col:
        # Create placeholder for vehicle log with a header
        st.markdown("### Vehicle Log")
        log_table = st.empty()
    
    return current_count, total_vehicles, avg_speed, log_table, video_frame, video_controls

def process_video(model, video_path, calib_data, tracker):
    SOURCE = np.array(calib_data['SOURCE'])
    TARGET_WIDTH = calib_data['TARGET_WIDTH']
    TARGET_HEIGHT = calib_data['TARGET_HEIGHT']
    
    TARGET = np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ])
    
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    polygon_zone = sv.PolygonZone(SOURCE)
    
    # Video input setup
    cap = cv2.VideoCapture(video_path)
    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(video_path)
    
    # Video output setup
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_info.fps,
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    # Initialize annotators
    thickness = 2
    text_scale = 1
    box_annotator = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, 
                                      text_position=sv.Position.BOTTOM_CENTER, 
                                      color_lookup=sv.ColorLookup.TRACK)
    
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    
    # Tracking configuration
    min_confidence = 0.5
    frame_count = 0
    
    # Create dashboard elements
    current_count, total_vehicles, avg_speed, log_table, video_frame, video_controls = create_dashboard()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Add stop button
    stop_button = video_controls.button("Stop Processing")
    
    update_interval = 5  # Update dashboard every 5 frames
    
    for frame in frame_generator:
        # Check if stop button was pressed
        if stop_button or not st.session_state.processing:
            st.session_state.processing = False
            break
            
        frame_count += 1
        current_time = datetime.now()
        
        # Update progress
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model(frame_rgb, verbose=False)[0]
        
        # Process detections
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter for vehicle classes only
        vehicle_mask = np.array([int(class_id) in VEHICLE_CLASSES.keys() 
                               for class_id in detections.class_id])
        confidence_mask = detections.confidence > min_confidence
        mask = vehicle_mask & confidence_mask
        
        detections = detections[mask]
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)
        
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)
        
        # Update current vehicle count for vehicles with enough data points
        visible_vehicles_with_speed = 0
        labels = []
        
        # Process each detection
        for tracker_id, class_id, [x, y] in zip(detections.tracker_id, detections.class_id, points):
            tracker_id = int(tracker_id)
            vehicle_type = VEHICLE_CLASSES[int(class_id)]
            coordinates[tracker_id].append(y)
            
            # Calculate speed if enough data points
            has_enough_data = len(coordinates[tracker_id]) >= video_info.fps / 2
            speed = None
            
            if has_enough_data:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time_delta = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time_delta * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")
                visible_vehicles_with_speed += 1
            else:
                labels.append(f"#{tracker_id} {vehicle_type}")
            
            # Update vehicle tracker only if we have enough data
            tracker.update_vehicle(tracker_id, speed, current_time, vehicle_type, has_enough_data)
        
        # Update current count to only include vehicles with enough data points
        tracker.current_count = visible_vehicles_with_speed
        
        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Update dashboard periodically
        if frame_count % update_interval == 0:
            # Update metrics
            current_count.metric("Current Vehicles", tracker.current_count)
            total_vehicles.metric("Total Vehicles", tracker.total_vehicles)
            
            # Calculate and update average speed
            all_speeds = [data['avg_speed'] for data in tracker.vehicle_log.values() if data['avg_speed'] > 0]
            overall_avg_speed = np.mean(all_speeds) if all_speeds else 0
            avg_speed.metric("Average Speed (km/h)", f"{overall_avg_speed:.1f}")
            
            # Update video frame
            video_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # Update vehicle log table
            log_table.dataframe(tracker.get_vehicle_data(), hide_index=True)
    
    # Release video writer
    out.release()
    cap.release()
    
    # Clear progress bar and status when stopped
    progress_bar.empty()
    status_text.empty()
    
    if not st.session_state.processing:
        st.warning("Processing stopped by user")
    else:
        # Store the output video path in session state
        st.session_state.output_video_path = output_path
        
        # Create a zip file containing both the video and JSON file
        zip_path = tempfile.NamedTemporaryFile(delete=False, suffix='.zip').name
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(output_path, arcname="processed_video.mp4")
            results = json.dumps(tracker.vehicle_log, indent=4, default=str)
            zipf.writestr("vehicle_log.json", results)
        
        # Replace stop button with a single download button for the zip file
        with video_controls:
            st.success("Processing completed!")
            st.download_button(
                label="Download Results",
                data=open(zip_path, 'rb').read(),
                file_name="results.zip",
                mime="application/zip"
            )
    
    return tracker.vehicle_log

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
        
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def main():
    st.set_page_config(page_title="Vehicle Speed Detection Dashboard", layout="wide")
    st.title("Vehicle Speed Detection Dashboard")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    video_file = st.sidebar.file_uploader("Upload video file", type=['mp4', 'avi', 'mov'])
    calib_file = st.sidebar.file_uploader("Upload calibration file (*.json)", type=['json'])
    
    if video_file and calib_file:
        # Save video file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name
            
        # Load calibration data
        calib_data = json.loads(calib_file.read())
        
        # Initialize vehicle tracker
        tracker = VehicleTracker()
        
        # Process button
        if st.sidebar.button("Start Processing"):
            try:
                st.session_state.processing = True
                # Load model
                model = load_model()
                
                # Process video
                vehicle_log = process_video(model, video_path, calib_data, tracker)
                
                # Clean up
                Path(video_path).unlink()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                st.sidebar.error(f"An error occurred: {str(e)}")
            finally:
                st.session_state.processing = False
                
                # Clean up output video if it exists
                if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
                    Path(st.session_state.output_video_path).unlink()
                    st.session_state.output_video_path = None
                
if __name__ == "__main__":
    main()