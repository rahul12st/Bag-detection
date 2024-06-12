import streamlit as st
from ultralytics import YOLO, solutions
from supervision.utils.video import VideoInfo, VideoSink, get_video_frames_generator
import cv2
import base64
import numpy as np

# Assuming CLASS_ID_BAG is defined somewhere in your code
CLASS_ID_BAG = 0
LINE_POSITION = 0.85  # Adjust this value as per your requirement

class Model:
    def __init__(self, variant='best.pt'):  # Default to your custom model
        self.model = YOLO('models/' + variant)
        self.CLASS_NAMES_DICT = self.model.model.names

    def predict_video_from_bytes(self, video_bytes: bytes, confidence_threshold: float = 0.9):
        video_frames = VideoInfo.from_video_bytes(video_bytes)
        total_frames = len(video_frames)
        current_frame = 0
        progress_bar = st.progress(0)

        line_y = int(video_frames[0].height * LINE_POSITION)
        line_points = [(0, line_y), (video_frames[0].width, line_y)]
        
        counter = solutions.ObjectCounter(
            reg_pts=line_points,  
            classes_names=self.CLASS_NAMES_DICT,
            draw_tracks=True,
            line_thickness=2,
        )

        output_frames = []

        for frame in get_video_frames_generator(video_frames):
            current_frame += 1
            progress_text = f'Frames: {current_frame}/{total_frames}, {round(100*current_frame/total_frames, 1)}% | The video is being processed!'
            progress_bar.progress(current_frame/total_frames, 'Completed!' if current_frame == total_frames else progress_text)
            tracks = self.model.track(frame, persist=True, show=False)
            
            mask = tracks[0].boxes.conf >= confidence_threshold
            filtered_tracks = [track[mask] for track in tracks]
            # Debug: Print the number of detected objects in each frame
            
            print(f'Detected objects in frame {current_frame}: {len(tracks[0].boxes)}')

            frame = counter.start_counting(frame, filtered_tracks)
            
            # Draw the line on the frame
            cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 2)

            # Convert the frame to bytes
            _, encoded_frame = cv2.imencode('.jpg', frame)
            output_frames.append(encoded_frame.tobytes())

        return output_frames

def main():
    st.title("Object Detection with YOLOv8")
    st.write("This is a demo of object detection using a custom YOLOv8 model.")
    
    model = Model(variant='best.pt')  # Use your custom model
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    if video_file:
        video_bytes = video_file.read()
        st.write("Processing video...")
        processed_video_frames = model.predict_video_from_bytes(video_bytes)
        
        st.write("Video processing completed. Download the processed video:")
        for idx, frame_bytes in enumerate(processed_video_frames):
            b64 = base64.b64encode(frame_bytes).decode()
            href = f'<a href="data:image/jpeg;base64,{b64}" download="processed_video_frame_{idx}.jpg">Download Frame {idx}</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
