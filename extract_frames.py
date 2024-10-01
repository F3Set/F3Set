import json
import os
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')

# save images
def save_imgs(cap, start, end, file_name, dim=224):
    if not os.path.isdir('./vid_frames_%d' % dim):
        os.mkdir('./vid_frames_%d' % dim)
    path = os.path.join('./vid_frames_%d' % dim, file_name)
    if not os.path.isdir(path):
        os.mkdir(path)
    count = 0
    cap.set(1, start)
    for t in range(start, end):
        frame_path = './vid_frames_%d/%s/%06d.jpg' % (dim, file_name, count)
        # if not os.path.exists(frame_path):
        _, frame = cap.read()
        H, W, _ = frame.shape
        resized = cv2.resize(frame, (W * dim // H, dim))
        cv2.imwrite(frame_path, resized)
        count += 1

data_name = ['train', 'val', 'test']
dim = 224
for i, name in enumerate(data_name):
    print(name)
    out = []
    json_file = json.load(open('./data/f3tennis/%s.json' % name))
    for clip in tqdm(json_file):
        match_id = '_'.join(clip['video'].split('_')[:-2])
        video_name = './videos/%s.mp4' % match_id
        cap = cv2.VideoCapture(video_name)
        # save images
        start, end = int(clip['video'].split('_')[-2]), int(clip['video'].split('_')[-1])
        file_name = clip['video']
        save_imgs(cap, start, end, file_name, dim=dim)