import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

def show_video_frame(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f'Error: Unable to open video file: {video_path}')
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f'Error: Unable to read video frame: {video_path}')
        return None
        
    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(current_frame, (1280, 720))
    

def get_current_frame(video, frame_index):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame, (1280, 720))
    else:
        return None

def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return video, total_frames

def get_dimensions(video):
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height

def scale_video(video, new_size):
    width, height = get_dimensions(video)
    x_scale = new_size[0] / width
    y_scale = new_size[1] / height
    return x_scale, y_scale