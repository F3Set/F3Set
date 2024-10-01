import os

def get_video_directories(path):
    data_path = os.path.join(path, "videos")
    return [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(path, "videos", d))]

def get_video_files(path, directory):
    if not directory:
        return []
    data_path = os.path.join(path, "videos")
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    files = [f for f in os.listdir(os.path.join(data_path, directory)) 
            if os.path.isfile(os.path.join(data_path, directory, f)) 
            and os.path.splitext(f)[1].lower() in video_extensions]
    
    return files