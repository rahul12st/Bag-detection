import streamlit as st
from models import Model
from utils import load_yaml_as_dict, save_file

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
    model.predict_video(
        source=PATHS['SOURCES'] + upload.name,
        target=PATHS['OUTPUTS'] + 'detected - ' + upload.name
    )
    