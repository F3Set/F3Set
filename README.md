# $F^3Tennis$: A Dataset for Analyzing Fast, Frequent, and Fine-grained Event Sequences from Videos
## Overview
Analyzing Fast, Frequent, and Fine-grained ($F^3$) events presents a significant challenge in video analytics and multi-modal LLMs. Although current methods exhibit efficacy on public benchmarks, they struggle to detect and identify $F^3$ events accurately due to challenges such as motion blur and subtle visual discrepancies. To address this, we introduce $F^3Tennis$, a new benchmark dataset built on tennis video specifically for $F^3$ event detection. $F^3Tennis$ is characterized by its extensive scale and comprehensive detail, encompassing over 1,000 event types with precise timestamps and supporting multi-level granularity. We evaluated popular temporal action understanding methods on $F^3Tennis$, revealing substantial challenges for existing techniques. Using tennis as a case study, we demonstrate the utility of $F^3$ sequences for advanced automated strategic analytics. The dataset and the benchmark code are accessible at https://github.com/F3EST/F3Tennis.

## Environment
The code is tested in Linux (Ubuntu 22.04) with the dependency versions in requirements.txt.

## Dataset
Refer to the READMEs in the [data](https://github.com/F3EST/F3Tennis/tree/main/data) directory for pre-processing and setup instructions.

## Basic usage
To train all baseline models, use `python3 train_f3tennis_baselines.py <dataset_name> <frame_dir> -s <save_dir> -m <model_arch> -t <head_arch>`.

* `<dataset_name>`: f3tennis
* `<frame_dir>`: path to the extracted frames
* `<save_dir>`: path to save logs, checkpoints, and inference results
* `<model_arch>`: feature extractor architecture (e.g., rny002, rny002_tsm, slowfast)
* `<head_arch>`: head module architecture (e.g., mstcn, asformer, gcn, actionformer, gru)

Similarly, we also provide the code `train_f3tennis_multilabel.py` for training under the multi-label setting. Use `python3 train_f3tennis_multilabel.py <dataset_name> <frame_dir> -s <save_dir> -m <model_arch> -t <head_arch>`. However, for the `<head_arch>` in `train_f3tennis_multilabel.py`, currently we only implemented `gru`. 

Training will produce checkpoints, predictions for the `val` split, and predictions for the `test` split on the best validation epoch.

### Trained models
Models and configurations can be found in [f3tennis-model](https://github.com/F3EST/F3Tennis/tree/main/f3tennis-model). Place the checkpoint file and config.json file in the same directory.

To perform inference with an already trained model, use `python3 test_f3tennis_baselines.py <model_dir> <frame_dir> -s <split>` or `python3 test_f3tennis_multilabel.py <model_dir> <frame_dir> -s <split>`. This will output results for 3 evaluation metrics (event-wise mean F1 score, element-wise mean F1 score, and edit score).

## Data format
Each dataset has plaintext files that contain the list of event types `events.txt` and elements: `elements.txt`

This is a list of the event names, one per line: `{split}.json`

This file contains entries for each video and its contained events.
```
[
    {
        "fps": 25,
        "height": 720,
        "width": 1280,
        "num_frames": 342,  // number of frames in this clip
        "video": "20210909-W-US_Open-SF-Aryna_Sabalenka-Leylah_Fernandez_170943_171285",  // "video name"_"start frame of the clip"_"end frame of the clip"
        "far_name": "Leylah Fernandez",  // far-end player's name
        "far_hand": "LH",  // far-end player's handedness
        "far_set": 1,  // far-end player's set score
        "far_game": 2,  // far-end player's game score
        "far_point": 2,  // far-end player's point score
        "near_name": "Aryna Sabalenka",  // near-end player's name
        "near_hand": "RH",  // near-end player's handedness
        "near_set": 1,  // near-end player's set score
        "near_game": 2,  // near-end player's game score
        "near_point": 0,  // near-end player's point score
        "events": [
            {
                "frame": 100,               // Frame
                "label": EVENT_NAME,        // Event type
            },
            ...
        ],
    },
    ...
]
```
**Frame directory**

We assume pre-extracted frames, that have been resized to 224 pixels high or similar. The organization of the frames is expected to be <frame_dir>/<video_id>/<frame_number>.jpg. For example,
```
video1/
├─ 000000.jpg
├─ 000001.jpg
├─ 000002.jpg
├─ ...
video2/
├─ 000000.jpg
├─ ...
```
Similar format applies to the frames containing objects of interest.

## Acknowledgement
This code base is largely from [E2E-Spot](https://github.com/jhong93/spot). Many thanks to the authors.








