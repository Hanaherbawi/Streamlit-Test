import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # You can add any image processing here if needed
        return frame  # Return the frame as is for now

def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Webcam Display Streamlit App")
    st.caption("Powered by OpenCV, Streamlit, and WebRTC")

    # Create a WebRTC streamer
    webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

if __name__ == "__main__":
    main()
