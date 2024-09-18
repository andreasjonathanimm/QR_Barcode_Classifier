import sys
import os
import csv
import av
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration
)

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

MODEL_PATH = 'models/'
MODEL_NAME = 'model'

CLASSSES = ['Barcode', 'Else', 'QR']
NUM_CLASSES = len(CLASSSES)

# Load model
try:
    MODEL = tf.saved_model.load(MODEL_PATH + '/' + MODEL_NAME)
except OSError:
    st.error('Model not found. Please train a model first.')
    sys.exit()

class VideoProcessor(object):
    def __init__(self):
        self._model = MODEL
        self._frame_count = 0
        self._last_frame = None
        self._last_frame_time = None
        self._last_frame_prediction = None
        self._last_frame_prediction_time = None
        self._image_width = IMAGE_WIDTH
        self._image_height = IMAGE_HEIGHT

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._frame_count += 1
        frame = frame.to_ndarray(format="bgr24")

        # Resize frame
        frame = cv2.resize(frame, (self._image_width, self._image_height))

        # Predict
        prediction = self._model.predict(np.array([frame]))
        prediction = CLASSSES[np.argmax(prediction)]

        # Output frame
        frame = cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        frame = cv2.resize(frame, (640, 480))

        return av.VideoFrame.from_ndarray(frame, format="bgr24")
    
    async def coroutine(self, frame):
        """Coroutine that yields processed frames"""
        return self.recv(frame)
    
    async def recv_queued(self, frames):
        """Coroutine that yields processed frames"""
        result = []
        for frame in frames[-5:]:
            result.append(await self.coroutine(frame))
        return result

def main():
    st.title("QR/Barcode Classifier")
    st.write("This app detects QR codes and barcodes in a video stream.")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_ctx = webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        st.write(webrtc_ctx.video_processor._frame_count)

if __name__ == "__main__":
    main()
