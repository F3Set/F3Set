import os
import gradio as gr
from utils.handle_video import load_video, show_video_frame
from utils.handle_directory import get_video_directories, get_video_files

class SelectDirectoryPage:
    def __init__(self, visible=True):
        self.select_directory_page, self.directory_dropdown, self.video_dropdown = self.build_select_directory_page(visible=visible)
        self.next_page = None
        self.next_page_button = None
    
    def build_select_directory_page(self, visible=True):
        select_directory_page = gr.Group(visible=visible)
        with select_directory_page:
            gr.Markdown("# Video Selector")
            directory_dropdown = gr.Dropdown(choices=get_video_directories("data"), label="Select Video Directory")
            video_dropdown = gr.Dropdown(choices=[], label="Select Video File")

            # update video files when directory is selected
            directory_dropdown.change(
                fn=self.update_video_files,
                inputs=[directory_dropdown, video_dropdown],
                outputs=[video_dropdown]
            )
                
        return select_directory_page, directory_dropdown, video_dropdown
    
    def build_label_net_page_button(self, label_net_page):
        select_button = gr.Button("Select Video")
        self.next_page_button = select_button
        self.next_page = label_net_page
        select_button.click(
            self.show_label_page, 
            inputs=[self.directory_dropdown, self.video_dropdown],
            outputs=[self.select_directory_page, label_net_page.label_net_page, label_net_page.selected_video_file, self.next_page_button, label_net_page.next_page_button, label_net_page.prev_page_button, label_net_page.frame, label_net_page.slider]
        )
        return select_button
        
    def update_video_files(self, directory_dropdown, video_dropdown):
        video_files = get_video_files("data", directory_dropdown)
        
        if not video_files:
            gr.Warning("No video files found in this directory.", duration=10)
            return gr.update(choices=[])
        
        return gr.update(choices=video_files)
    
    def show_label_page(self, directory, video):
        if not directory or not video:
            gr.Warning("Please select a valid directory and video file.")
            return gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=None)
        
        video_path = os.path.join("data", "videos", directory, video)
        frame = show_video_frame(video_path)
        self.next_page.video_path = video_path
        self.next_page.video, self.next_page.total_frames = load_video(video_path)        
        slider = gr.Slider(minimum=0, maximum=self.next_page.total_frames -1, step=1, value=1, label="Frame Slider")
         
        if frame is None:
            gr.Warning("Error loading video frame.")
            return gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=None), gr.update(value=None)
        else:
            # directory_page, label_net_page, video_directory, next_page_button, label_net_prev_page_button, frame
            return gr.update(visible=False), gr.update(visible=True), gr.update(value=video_path), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(value=frame), slider
    