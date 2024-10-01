import os
import cv2
import json
import gradio as gr
from PIL import Image, ImageDraw
from utils.handle_video import show_video_frame, load_video, get_current_frame, scale_video

class LabelNetPage:
    def __init__(self, visible=True):
        self.next_page = None
        self.next_page_button = None
        self.prev_page = None
        self.prev_page_button = None
        self.video_path = None
        self.net = [-1, -1]  # net position for new video        
        self.total_frames = None
        self.video = None
        self.label_net_page, self.selected_video_file, self.frame, self.slider = self.build_label_net_page(visible=visible)
    
    def build_label_net_page(self, visible=True):
        label_net_page = gr.Group(visible=visible)
        with label_net_page:
            
            # Labelling Instructions
            instructions = gr.Markdown("""# Label Net Page
                        Instructions: Click on the image to select the middle bottom of the tennis net, then press 'Confirm Net Position
                        """)
            selected_video_file = gr.Textbox(label="Selected Video File", value=None, interactive=False)
            
            # Video frame and slider
            frame = gr.Image(label="Video Frame")
            coords_output = gr.Textbox(label="Coordinates", value=self.net)
            slider = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Frame Slider")
            
            
            slider.release(self.update_frame, inputs=[slider], outputs=[frame, slider]) # set up slider to update frame
            frame.select(
                self.get_click_coordinates,
                inputs=[slider],
                outputs=[coords_output, frame]
            )
            
            # Navigation buttons
            self.next_page_button = gr.Button("Confirm Net Position")
            self.prev_page_button = gr.Button("Back to Select Video", visible=False)
                
        return label_net_page, selected_video_file, frame, slider
    
    def update_frame(self, slider):
        frame = get_current_frame(self.video, slider)
        return frame, slider

    def setup_prev_page_button(self, select_directory_page):
        self.prev_page = select_directory_page
        self.prev_page_button.click(
            self.show_select_directory_page, 
            inputs=[],
            outputs=[select_directory_page.select_directory_page, select_directory_page.next_page_button, self.label_net_page, self.prev_page_button, self.next_page_button]
        )

    def show_select_directory_page(self):
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    def setup_next_page_button(self, label_page):
        self.next_page = label_page
        label_page.net = self.net
        self.next_page_button.click(
            self.show_label_page, 
            inputs=[],
            # initialize slider and first video frame too
            outputs=[self.label_net_page, self.prev_page_button, self.next_page_button, label_page.label_page, 
                     label_page.prev_page_button, label_page.current_frame, label_page.slider, label_page.event_list] + label_page.players
        )

    def show_label_page(self):
        # handle net = [-1, -1]
        if self.net[0] == -1 or self.net[1] == -1:
            gr.Warning("Please select a valid net position.")
            return None, None, None, None
        
        video, total_frames = load_video(self.video_path)        
        current_frame = get_current_frame(video, 0)
        self.next_page.video, self.next_page.total_frames = video, total_frames
        # self.next_page.net = self.scale_net_position(video, (1280, 720))
        self.next_page.net = self.net
        event_list, p1, p2, p3, p4 = self.next_page.load_event_list(self.video_path)
        
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), 
                gr.update(visible=True), current_frame, gr.Slider(minimum=0, maximum=total_frames, step=1, value=1, label="Frame Slider"), 
                event_list, p1, p2, p3, p4)

    def get_click_coordinates(self, slider, evt: gr.SelectData):
        try:
            x, y = evt.index
            self.net = [x, y]            
            img = self.draw_dot(slider, x, y)

            return gr.update(value=self.net), img
        
        except Exception as e:
            gr.Warning(f"Error occured while processing: {e}")
            return None, self.frame
    
    def draw_dot(self, slider, x, y):
        img = get_current_frame(self.video, slider)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.ellipse([x-5, y-5, x+5, y+5], fill='blue', outline='blue')
        return img
    
    def scale_net_position(self, video, new_size):
        original_net = self.net
        x_scale, y_scale = scale_video(video, new_size)
        return [int(original_net[0] * x_scale), int(original_net[1] * y_scale)]
    
    
    # def build_label_page_button(self, label_page):
    #     confirm_net_button = gr.Button("Confirm Net Position")
    #     self.next_page_button = confirm_net_button
    #     self.next_page = label_page
    #     confirm_net_button.click(
    #         self.show_label_page, 
    #         inputs=[],
    #         outputs=[self.label_net_page, label_page.label_page]
    #     )
    #     return select_button

    # def show_label_page(self):
    #     self.next_page.net = self.net
    #     return gr.update(visible=False), gr.update(visible=True)
        