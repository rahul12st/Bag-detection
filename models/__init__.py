from ultralytics import YOLO, solutions
import cv2
from supervision.utils.video import VideoInfo, VideoSink, get_video_frames_generator
from supervision.detection.core import Detections
from utils import create_colorpalette, hex_to_rgb
import streamlit as st
import numpy as np

colors = [
    "#a351fb", "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    "#d2f53c", "#fabebe", "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000", "#aaffc3",
]

PATHS = {
    'SOURCES': '/path/to/sources/',
    'OUTPUTS': '/path/to/outputs/',
    # Other paths...
} 

# Assuming CLASS_ID_BAG is defined somewhere in your code
CLASS_ID_BAG = 0
LINE_POSITION = 0.85  # Adjust this value as per your requirement

class Model:
    def __init__(self, variant='best.pt'):  # Default to your custom model
        self.model = YOLO('models/' + variant)
        self.CLASS_NAMES_DICT = self.model.model.names

    def predict_video(self, source: str, target: str, confidence_threshold: float = 0.9):
        generator = get_video_frames_generator(source)
        
        video_info = VideoInfo.from_video_path(source)
        total, current = video_info.total_frames, 0
        progress_text = f'Frames: {current}/{total}, {round(100*current/total, 1)}% | The video is being processed!'
        progress_bar = st.progress(current/total, progress_text)
       
        line_y = int(video_info.height * LINE_POSITION)
        line_points = [(0, line_y), (video_info.width, line_y)]
        
        counter = solutions.ObjectCounter(
            reg_pts=line_points,  
            classes_names=self.CLASS_NAMES_DICT,
            draw_tracks=True,
            line_thickness=2,
        )

        with VideoSink(target, video_info) as sink:
            for frame in generator:
                progress_text = f'Frames: {current}/{total}, {round(100*current/total, 1)}% | The video is being processed!'
                current += 1
                progress_bar.progress(current/total, 'Completed!' if current == total else progress_text)
                tracks = self.model.track(frame, persist=True, show=False)
                
                mask = tracks[0].boxes.conf >= confidence_threshold
                filtered_tracks = [track[mask] for track in tracks]
                # Debug: Print the number of detected objects in each frame
                
                print(f'Detected objects in frame {current}: {len(tracks[0].boxes)}')

                frame = counter.start_counting(frame, filtered_tracks)
                
                # Draw the line on the frame
                cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 2)
                
                sink.write_frame(frame)
        
        
       
if __name__ == "__main__":
    st.title("Object Detection with YOLOv8")
    st.write("This is a demo of object detection using a custom YOLOv8 model.")
    
    model = Model(variant='best.pt')  # Use your custom model
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    if video_file:
        with open(PATHS['SOURCES'] + video_file.name, "wb") as f:
            f.write(video_file.getbuffer())
        model.predict_video(PATHS['SOURCES'] + video_file.name, PATHS['OUTPUTS'] + "output_" + video_file.name)
