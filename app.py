import tempfile
import streamlit as st
from models import Model
from utils import load_yaml_as_dict, save_file
import base64
import os

# Load variables and paths
VARIABLES = load_yaml_as_dict('variables/variables.yaml')
PATHS = VARIABLES['PATHS']

# Ensure directories exist
os.makedirs(PATHS['SOURCES'], exist_ok=True)
os.makedirs(PATHS['OUTPUTS'], exist_ok=True)

# Initialize the model with the 'best.pt' variant directly
model = Model(variant='best.pt')

# File uploader for video files
upload = st.file_uploader(
    'Upload a video or an image!',
    type=['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
)

if upload is not None:
    # Use a temporary file for the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(upload.getbuffer())
        source_path = tmpfile.name

    if st.button('Start detecting'):
        processed_video_path = model.predict_video(source=source_path)
        
        st.write("Video processing completed. Download the processed video:")
        with open(processed_video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            b64 = base64.b64encode(video_bytes).decode()
            href = f'<a href="data:file/mp4;base64,{b64}" download="processed_video.mp4">Download Processed Video</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        os.remove(processed_video_path)  # Remove temporary processed video file
        os.remove(source_path)  # Remove temporary source video file
