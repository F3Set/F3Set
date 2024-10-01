def build_label_net_page_button(self, label_net_page):
        select_button = gr.Button("Select Video")
        self.next_page = select_button
        select_button.click(
            self.show_label_page, 
            inputs=[self.directory_dropdown, self.video_dropdown],
            outputs=[self.select_directory_page, label_net_page.label_net_page, label_net_page.selected_video_file, self.next_page, label_net_page.prev_page, label_net_page.frame]
        )
        return select_button

def show_label_page(self, directory, video):
        if not directory or not video:
            gr.Warning("Please select a valid directory and video file.")
            return gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(visible=True), gr.update(visible=False), gr.update(value=None)
        
        video_path = os.path.join("data", "videos", directory, video)
        frame = show_video_frame(video_path)
         
        if frame is None:
            gr.Warning("Error loading video frame.")
            return gr.update(visible=True), gr.update(visible=False), gr.update(value=None), gr.update(visible=True), gr.update(visible=False), gr.update(value=None)
        else:
            # directory_page, label_net_page, video_directory, next_page_button, label_net_prev_page_button, frame
            return gr.update(visible=False), gr.update(visible=True), gr.update(value=video_path), gr.update(visible=False), gr.update(visible=True), gr.update(value=frame)