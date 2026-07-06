import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import sys
import os

from live_demo import load_mediapipe_model, generate_tensor_from_frames, load_attention_extractor_v2, extract_landmarks

def main():
    detector = load_mediapipe_model()
    
    cap = cv2.VideoCapture("dataset/raw_videos/session_001.mp4")
    frames = []
    for _ in range(7):
        ret, frame = cap.read()
        if ret: frames.append(frame)
    cap.release()
    
    print(f"Read {len(frames)} frames")
    tensor = generate_tensor_from_frames(frames, detector)
    print("Tensor shape:", tensor.shape)
    print("Tensor mean:", np.mean(tensor))
    print("Tensor min/max:", np.min(tensor), np.max(tensor))
    
    # Predict
    extractor, att_layer = load_attention_extractor_v2()
    X_input = np.expand_dims(tensor, axis=0)
    scores, bilstm_out = extractor.predict(X_input, verbose=0)
    print("Scores:", scores)
    
if __name__ == "__main__":
    main()
