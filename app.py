import streamlit as st
from models import Model
import base64

# Initialize the model with the 'best.pt' variant directly
model = Model(variant='best.pt')

# File uploader for video files
upload = st.file_uploader(
    'Upload a video or an image!',
    type=['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
)

if upload is not None:
    if st.button('Start detecting'):
        # Save the uploaded file in memory
        upload_bytes = upload.getvalue()

        # Process the video directly from memory
        processed_video_bytes = model.predict_video_from_bytes(upload_bytes)
        
        st.write("Video processing completed. Download the processed video:")
        href = f'<a href="data:file/mp4;base64,{base64.b64encode(processed_video_bytes).decode()}" download="processed_video.mp4">Download Processed Video</a>'
        st.markdown(href, unsafe_allow_html=True)
