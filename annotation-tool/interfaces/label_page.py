import os
import cv2
import json
import gradio as gr
from utils.handle_video import show_video_frame, get_current_frame, scale_video

class LabelPage:
    def __init__(self, visible=True):
        self.prev_page = None
        self.prev_page_button = None
        self.video = None
        self.total_frames = None
        self.net = None
        self.events = []
        self.player_names = {"p1": None, "p2": None, "p3": None, "p4": None}
        self.label_page, self.current_frame, self.slider, self.label_button, self.delete_button, self.event_list, self.labels, self.players = self.label_page(visible=visible)
        
    def label_page(self, visible=True):
        label_page = gr.Group(visible=visible)
        with label_page:
            with gr.Row(): gr.Markdown(""" ## Player Description""", elem_classes="column-container")
            
            # Player description
            with gr.Row():
                with gr.Column(elem_classes="column-container"):
                    p1 = gr.Textbox(label="Player 1", placeholder="Enter P1 Description", interactive=True)
                    p2 = gr.Textbox(label="Player 2", placeholder="Enter P2 Description", interactive=True)
                
                with gr.Column(elem_classes="column-container"):
                    p3 = gr.Textbox(label="Player 3", placeholder="Enter P3 Description", interactive=True)
                    p4 = gr.Textbox(label="Player 4", placeholder="Enter P4 Description", interactive=True)
                
            players = [p1, p2, p3, p4]
    
            # Labeling Information
            with gr.Row():
                with gr.Column(scale=2, elem_classes="column-container"):
                    with gr.Row():
                        current_frame = gr.Image(label="Current Frame", height=720, width=1280)
                        
                    with gr.Row():
                        slider = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Frame Slider")
                    
                    # Forward & Backward navigations
                    with gr.Row():
                        skip_back_10_frames = gr.Button("← 10 frames")
                        skip_back_5_frames = gr.Button("← 5 frames")
                        skip_back_1_frame = gr.Button("← 1 frames")
                        skip_1_frame = gr.Button("1 frames →")
                        skip_5_frames = gr.Button("5 frames →")
                        skip_10_frames = gr.Button("10 frames →")
                        
                    with gr.Row():
                        skip_back_10s = gr.Button("← 10s")
                        skip_back_5s = gr.Button("← 5s")
                        skip_back_1s = gr.Button("← 1s")
                        skip_1s = gr.Button("1s →")
                        skip_5s = gr.Button("5s →")
                        skip_10s = gr.Button("10s →")
                        
                with gr.Column(scale=1, elem_classes="column-container"):
                    with gr.Row(elem_id="row1"):
                        gr.Markdown(""" ## Event Labeling""")
                    with gr.Row():
                        labeled_frame_number = gr.Number(label="Frame Number", value=1, visible=True, interactive=False)
                    with gr.Row():
                        player = gr.Radio(["P1", "P2", "P3", "P4"], label="Player", interactive=True)
                    with gr.Row():
                        court_position = gr.Radio(["Far deuce", "Far ad", "Near ad", "Near deuce"], label="Court Position", interactive=True)
                    with gr.Row():
                        side = gr.Radio(["Forehand", "Backhand"], label="Side", interactive=True)
                    with gr.Row():
                        shot_type = gr.Radio(["Serve", "Return", "Volley", "Lob", "Smash", "Swing"], label="Shot Type", interactive=True)
                    with gr.Row():
                        shot_direction = gr.Radio(["T", "B", "W", "CC", "DL", "DM", "II", "IO"], label="Shot Direction", interactive=True)
                    with gr.Row():
                        formation = gr.Radio(["Conventional", "I-formation", "Australian", "Non-serve"], label="Formation", interactive=True)
                    with gr.Row():
                        outcome = gr.Radio(["In", "Win", "Err"], label="Outcome", interactive=True)
                    with gr.Row():
                        player_coordinates = gr.Textbox(label="Player Coordinates", visible=False)
                    with gr.Row(elem_id="row3", elem_classes="column-container"):
                        gr.Markdown("")
                    with gr.Row():
                        label_button = gr.Button("Label Event")
                    with gr.Row():
                        delete_button = gr.Button("Delete Frame")
                    with gr.Row():
                        save_button = gr.Button("Save Labels")
                    
                    # Store labels here
                    labels = [player, court_position, side, shot_type, shot_direction, formation, outcome, player_coordinates, labeled_frame_number]
            
            event_list = gr.Code(value=None, label="Labeled Events", language="json", interactive=False)
            save_status = gr.Textbox(label="Save Status", value="Not Saved")
            self.prev_page_button = gr.Button("Back to Net", visible=False)
            
            # Player Description Update
            for i, player in enumerate(players):
                    player.change(self.update_player, inputs=[player, gr.Number(value=i+1, visible=False)], outputs=[event_list, save_status])
            
            # Slider update
            slider.release(self.update_frame, inputs=[slider], outputs=[current_frame, slider]) # set up slider to update frame
            
            # Backward navigation
            skip_back_1_frame.click(self.skip_frames, inputs = [gr.Number(-1, visible=False), slider], outputs=[current_frame, slider])
            skip_back_5_frames.click(self.skip_frames, inputs = [gr.Number(-5, visible=False), slider], outputs=[current_frame, slider])
            skip_back_10_frames.click(self.skip_frames, inputs = [gr.Number(-10, visible=False), slider], outputs=[current_frame, slider])
            skip_back_1s.click(self.skip_seconds, inputs = [gr.Number(-1, visible=False), slider], outputs=[current_frame, slider])
            skip_back_5s.click(self.skip_seconds, inputs = [gr.Number(-5, visible=False), slider], outputs=[current_frame, slider])
            skip_back_10s.click(self.skip_seconds, inputs = [gr.Number(-10, visible=False), slider], outputs=[current_frame, slider])

            # Forward navigation
            skip_1_frame.click(self.skip_frames, inputs = [gr.Number(1, visible=False), slider], outputs=[current_frame, slider])
            skip_5_frames.click(self.skip_frames, inputs = [gr.Number(5, visible=False), slider], outputs=[current_frame, slider])
            skip_10_frames.click(self.skip_frames, inputs = [gr.Number(10, visible=False), slider], outputs=[current_frame, slider])
            skip_1s.click(self.skip_seconds, inputs = [gr.Number(1, visible=False), slider], outputs=[current_frame, slider])
            skip_5s.click(self.skip_seconds, inputs = [gr.Number(5, visible=False), slider], outputs=[current_frame, slider])
            skip_10s.click(self.skip_seconds, inputs = [gr.Number(10, visible=False), slider], outputs=[current_frame, slider])
            
            # Labeling
            current_frame.select(self.handle_image_click, inputs=[slider], outputs=[court_position, player_coordinates, labels[-1]])
            label_button.click(self.label_event, inputs=labels, outputs=[event_list, save_status] + labels)
            save_button.click(self.save_labels, inputs=[event_list], outputs=[save_status])
            delete_button.click(self.delete_event, inputs=[slider], outputs=[event_list, save_status])
            
        return label_page, current_frame, slider, label_button, delete_button, event_list, labels, players

    def setup_prev_page_button(self, label_net_page):
        self.prev_page = label_net_page
        self.prev_page_button.click(
            self.show_label_net_page, 
            inputs=[],
            outputs=[self.label_page, self.prev_page_button, label_net_page.label_net_page, label_net_page.prev_page_button, label_net_page.next_page_button]
        )
    
    def show_label_net_page(self):
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
    
    def update_frame(self, slider):
        frame = get_current_frame(self.video, slider)
        return frame, slider

    def skip_frames(self, num_frames, slider):
        try:
            new_frame_index = max(1, min(slider + num_frames, self.total_frames))
            return self.update_frame(new_frame_index)
        except Exception as e:
            gr.Warning(f"Encountered an error while skipping frames: {e}")

    def skip_seconds(self, num_seconds, slider):
        try:
            fps = self.video.get(cv2.CAP_PROP_FPS)
            return self.skip_frames(int(num_seconds * fps), slider)

        except Exception as e:
            gr.Warning(f"Encountered an error while skipping frames: {e}")
    
    def label_event(self, player, court_position, side, shot_type, shot_direction, formation, outcome, player_coordinates, labeled_frame_number):
        if not player: gr.Warning("Please select a player."); return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if not court_position: gr.Warning("Please select a court position."); return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if not side: gr.Warning("Please select a side."); return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if not shot_type: gr.Warning("Please select a shot type."); return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if not shot_direction: gr.Warning("Please select a shot direction."); return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if not formation: gr.Warning("Please select a formation."); return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if not outcome: gr.Warning("Please select an outcome."); return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if not player_coordinates or labeled_frame_number is None: gr.Warning("Please click the player on the image (hit coordinates)"); return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if formation != 'Non-serve' and shot_type != 'Serve': gr.Warning("Formation should be 'Non-serve' for non-serve shots."); return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if formation == 'Non-serve' and shot_type == 'Serve': gr.Warning("Formation should not be 'Non-serve' for serve shots."); return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if shot_type == 'Serve' and shot_direction not in ['W', 'T', 'B']: gr.Warning("Invalid shot direction for serve."); return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if not self.valid_court_side_direction(court_position, side, shot_direction): gr.Warning("Invalid combination of court position, side, and shot direction. Please re-label. (unless player is left-handed)")
        
        coarse_label = f"{player}_{court_position.lower().replace(' ', '_')}_{side.lower()}_{shot_type.lower()}_{shot_direction.lower()}_{formation.lower()}_{outcome.lower()}"
    
        # Check if there's an existing event for this frame
        existing_event_index = next((index for (index, d) in enumerate(self.events) if d["frame"] == labeled_frame_number), None)
    
        if existing_event_index is not None:
            # Update existing event
            self.events[existing_event_index] = {
                "frame": int(labeled_frame_number),
                "event": coarse_label,
                "relative_player_width": eval(player_coordinates)[0]/1280,
                "relative_player_height": eval(player_coordinates)[1]/720,
            }
        else:
            # Add new event
            self.events.append({
                "frame": int(labeled_frame_number),
                "event": coarse_label,
                "relative_player_width": eval(player_coordinates)[0]/1280,
                "relative_player_height": eval(player_coordinates)[1]/720,
            })
    
        # Sort events by frame number
        self.events.sort(key=lambda x: int(x["frame"]))
        
        current_video_id = self.prev_page.video_path.split("/")[-1].split(".")[0]
        event_update, status_update = self.update_event_list()
        return (event_update, status_update, 
                gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), 
                gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None))

    def update_event_list(self):
        current_video_id = self.prev_page.video_path.split("/")[-1].split(".")[0]
        events_json = json.dumps({"video_id": current_video_id, 
                                  "total_frames": self.total_frames, 
                                  "player_descriptions": self.player_names,
                                  "events": self.events}, indent=2)
        return gr.update(value=events_json, language="json"), gr.update(value="Labelled events not updated")
    
    def delete_event(self, slider):
        # Check for event index
        print(f'Deleting: frame {slider}')
        existing_event_index = next((index for (index, d) in enumerate(self.events) if d["frame"] == int(slider)), None)
        if existing_event_index is None: gr.Warning(f"Frame {slider} is not labelled"); return gr.update(), gr.update()
        self.events.pop(existing_event_index)
        return self.update_event_list()
        

    def save_labels(self, event_list):
        if not event_list: gr.Warning("No events to save."); return None
        current_video_id = self.prev_page.video_path.split("/")[-1].split(".")[0]
        os.makedirs("labelled", exist_ok=True)
        file_path = os.path.join("data", "labelled", f"{current_video_id}.json")
        
        with open(file_path, 'w') as f:
            json.dump(json.loads(event_list), f, indent=2)
        
        return gr.update(value=f"Saved to {file_path}")
    
    def load_event_list(self, video_path):
        try:
            current_video_id = video_path.split("/")[-1].split(".")[0]
            os.makedirs("data/labelled", exist_ok=True)
            file_path = os.path.join("data", "labelled", f"{current_video_id}.json")
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {"video_id": current_video_id, "total_frames": self.total_frames, "player_descriptions": self.player_names, "events": []}
            
            self.events = existing_data["events"]
            self.player_names = existing_data["player_descriptions"]
            return (gr.update(value=json.dumps(existing_data, indent=2), language="json"), gr.update(value=existing_data["player_descriptions"]["p1"]), 
                    gr.update(value=existing_data["player_descriptions"]["p2"]), gr.update(value=existing_data["player_descriptions"]["p3"]), 
                    gr.update(value=existing_data["player_descriptions"]["p4"]))
        
        except Exception as e:
            gr.Warning(f"Error loading event list: {e}")
            return gr.update(value=None)
    
    def update_player(self, player, i):
        self.player_names[f"p{i}"] = player
        return self.update_event_list()
    
    def get_court_position(self, x: int, y: int) -> str:
        if y < self.net[1]:
            return 'Far deuce' if x < self.net[0] else 'Far ad'
        else:
            return 'Near ad' if x < self.net[0] else 'Near deuce'
        
    def handle_image_click(self, slider, evt: gr.SelectData):
        try:
            if evt is None or evt.index is None:
                gr.Warning("Please click on the video frame.")
                return None, None
            x, y = evt.index
            court_pos = self.get_court_position(x, y)
            return gr.update(value=court_pos), gr.update(value=[x, y]), gr.update(value=slider)
        
        except Exception as e:
            gr.Warning(f"Error occured while processing: {e}")
            return None, None
        
    def valid_court_side_direction(self, court_position, side, shot_direction):
        if 'ad' in court_position.lower() and side == 'Forehand' and shot_direction in ['DL', 'CC']: return False
        if 'deuce' in court_position.lower() and side == 'Backhand' and shot_direction in ['DL', 'CC']: return False
        if 'ad' in court_position.lower() and side == 'Backhand' and shot_direction in ['II', 'IO']: return False
        if 'deuce' in court_position.lower() and side == 'Forehand' and shot_direction in ['II', 'IO']: return False
        return True