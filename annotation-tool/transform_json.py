from copy import deepcopy
from enum import Enum
import json
import argparse

RH_COMBINATIONS = {
    'deuce': {
        'forehand': ['dl', 'cc'],
        'backhand' : ['ii', 'io']
    },
    'ad': {
        'forehand' : ['ii', 'io'],
        'backhand' : ['dl', 'cc']
    }
}
TEMPLATE = {
        "video_id": None,
        "num_frames": None,
        "events":[]
    }  

class GameState(Enum):
    START = 0
    RETURN = 1
    STROKE = 2
    END = 3


def main(files):
    filepath = './data/labelled/'    
    for file in files:
        filename = file.split(".")[0]
        with open(f'{filepath}{file}', 'r') as json_data:
            d = json.load(json_data)  
            json_data.close()
            new_json_data = process(d)
        with open(f'{filepath}{filename}_transformed.json', 'w') as f:
            json.dump(new_json_data, f, indent=2)

def get_state(shot, outcome):
    '''
    args: Player_Side_Court_Handedness_Shot_Direction_Formation_Outcome
    '''            
    if "serve" in shot:
        return GameState.START
    if "return" in shot:
        return GameState.RETURN
    if "win" in outcome or "err" in outcome:
        return GameState.END        
    return GameState.STROKE

def handle_start(label, rally):
    '''
    This function takes in the empty json template, updates the start frame 
    in the video id, and adds the current event.
    '''
    current_frame = label['frame']
    rally['video_id'] += f'_{current_frame}'    
    rally['num_frames'] = label['frame']    
    label['frame'] = 0
    rally['events'].append(label)

def handle_normal(label, rally):
    label['frame'] -= int(rally['num_frames'])
    rally['events'].append(label)

def handle_end(label, rally):
    start_frame = rally['video_id'].split("_")[-1]
    assert(start_frame.isdigit())
    assert(rally['num_frames'])    
    end_frame = label['frame']
    label['frame'] -= int(rally['num_frames'])
    rally['num_frames'] = int(end_frame) - int(rally['num_frames'])
    rally['video_id'] += f'_{end_frame}'    
    rally['events'].append(label)

def process(json_data):     
    TEMPLATE['video_id'] = json_data['video_id']
    body = json_data['events']    
    results = []
    current_rally = deepcopy(TEMPLATE)    
    for label in body:        
        args = label['event'].split('_')
        _, _, court, hand, shot, direction, _, outcome = args
        state = get_state(shot, outcome)   
        if state != GameState.START and direction not in RH_COMBINATIONS[court][hand]:
            print(f"frame {label['frame']} Wrong Label")
            # return None             
        if state == GameState.START:
            handle_start(label, current_rally)                                  
        elif state == GameState.RETURN or state == GameState.STROKE:
            handle_normal(label, current_rally)        
        elif state == GameState.END:
            handle_end(label, current_rally)
            results.append(current_rally)            
            current_rally = deepcopy(TEMPLATE)   
        else: # added to ensure all cases are covered
            print(f"frame {label['frame']} Wrong Label")
            return None         
    return results
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-f", "--files", nargs='+', default=[])
    args = parser.parse_args()    
    main(args.files)
