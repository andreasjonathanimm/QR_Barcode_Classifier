"""Main file for running webcam to decode images and test model on test set.
Uses Streamlit and Streamlit-WebRTC."""

# Dependencies (pip3 install -r requirements.txt)
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
from old_code.model import load_dataset, load_model, predict, CLASSES, NUM_CLASSES

# Model path and name
MODEL_PATH = 'models/'
MODEL_NAME = 'model2'

# Test path
TEST_PATH = 'test'

# Load model
MODEL = load_model(MODEL_PATH + '/' + MODEL_NAME + '.h5')

def test():
    """Test model on test set and prints results
    
    Returns: None"""
    # Load and preprocess data
    x_test, y_test = load_dataset(TEST_PATH)
    x_test = x_test.astype('float32') / 255
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Evaluate model
    score = MODEL.evaluate(x_test, y_test, verbose=0)
    loss = score[0]
    accuracy = score[1]
    error = 1 - score[1]

    # Get predictions result for each image
    predictions = MODEL.predict(x_test)

    # Output results to file
    with open(MODEL_PATH + '/' + MODEL_NAME + '/test_results.txt', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Loss', 'Accuracy', 'Error'])
        writer.writerow([loss, accuracy, error])

        # Output predictions to file
        writer.writerow(['Image', 'Prediction', 'Actual'])
        for i, prediction in enumerate(predictions):
            writer.writerow([i, CLASSES[np.argmax(prediction)], CLASSES[np.argmax(y_test[i])]])

        print('Path: ' + MODEL_PATH + '/' + MODEL_NAME + '/test_results.txt')

class VideoProcessor(object):
    """Class for processing video frames"""
    def recv(self, frame):
        """Processes video frame and returns processed frame.
        frame: video frame
        
        Returns: processed video frame"""
        # Convert frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")

        # Get predictions
        prediction = predict(MODEL, img)

        # Get class with highest probability
        prediction = CLASSES[np.argmax(prediction)]

        # Draw prediction on frame
        img = cv2.putText(img, prediction, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Convert frame back to WebRTC format
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    async def coroutine(self, frame):
        """Processes video frame and returns a coroutine to process the frame.
        frame: video frame
        
        Returns: coroutine"""
        return self.recv(frame)

    async def recv_queued(self, frames):
        """Processes video frame coroutined and return processed frames.
        frames: video frames
        
        Returns: processed video frames"""
        result = []
        # Process only the last frame in the queue to reduce latency
        for frame in frames[-1:]:
            result.append(await self.coroutine(frame))
        return result

def main():
    """Main function, runs webcam to decode images
    
    Returns: None"""
    # Load model if it exists
    if not os.path.exists(MODEL_PATH + '/' + MODEL_NAME + '.h5'):
        print('Model ' + MODEL_NAME + ' not found in ' + MODEL_PATH +
            '.Please train model first or change model name.')
        exit()

    # Run webcam
    webrtc_ctx = webrtc_streamer(
        key="Webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers":
            [{"urls": ["stun:stun.l.google.com:19302"]}]},),
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test()
        elif sys.argv[1] == 'main':
            main()
        else:
            print('Invalid argument')
    else:
        print('Usage: python model.py (test|main)')
        print('test: test model on test set')
        print('main: run webcam to decode images')
        print('Streamlit: streamlit run model.py (main|test) to run in browser')
        st.write('Usage: streamlit run model.py (test|main)')
        st.write('test: test model on test set')
        st.write('main: run webcam to decode images')
