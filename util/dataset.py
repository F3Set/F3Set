import os

from util.io import load_text


DATASETS = [
    'f3set-tennis',
    'f3set-badminton',
    'f3set-table-tennis',
    'f3set-tennis-double',
    'shuttleset',
    'finediving',
    'finegym',
    'cctvpipe'
]


def load_classes(file_name):
    return {x: i + 1 for i, x in enumerate(load_text(file_name))}


def read_fps(video_frame_dir):
    with open(os.path.join(video_frame_dir, 'fps.txt')) as fp:
        return float(fp.read())


def get_num_frames(video_frame_dir):
    max_frame = -1
    for img_file in os.listdir(video_frame_dir):
        if img_file.endswith('.jpg'):
            frame = int(os.path.splitext(img_file)[0])
            max_frame = max(frame, max_frame)
    return max_frame + 1

