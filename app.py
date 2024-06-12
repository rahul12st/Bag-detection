import streamlit as st
from models import Model
from utils import load_yaml_as_dict, save_file
import base64
import os

# Load variables and paths
VARIABLES = load_yaml_as_dict('variables/variables.yaml')
PATHS = VARIABLES['PATHS']

# Initialize the model with the 'best.pt' variant directly
model = Model(variant='best.pt')

# File uploader for video files
upload = st.file_uploader(
    'Upload a video or an image!',
    type=['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
)

if upload is not None:
    save_file(upload, PATHS['SOURCES'] + upload.name)

if st.button('Start detecting'):
    processed_video_path = model.predict_video(source=PATHS['SOURCES'] + upload.name)
    
    st.write("Video processing completed. Download the processed video:")
    with open(processed_video_path, 'rb') as video_file:
        video_bytes = video_file.read()
        b64 = base64.b64encode(video_bytes).decode()
        href = f'<a href="data:file/mp4;base64,{b64}" download="processed_video.mp4">Download Processed Video</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    os.remove(processed_video_path)  # Remove temporary processed video file
