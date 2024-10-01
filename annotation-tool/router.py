import os
import gradio as gr
from interfaces import select_directory_page, label_net_page

class Router:
    def __init__(self, app):
        self.app = app
        self.pages = self.get_pages()
        self.current_page = None
        self.next_page = None
        self.prev_page = None
        self.navigate_select_directory_page()
        
    def get_pages(self):
        return {
            "select_directory_page": select_directory_page.SelectDirectoryPage(),
            "label_net_page": label_net_page.LabelNetPage(self)
        }
    
    def navigate(self, path):
        for page in self.pages:
            if page == path:
                print(f'Showing {page}')
                self.current_page = self.pages[page]
                self.current_page.visible = True
            else:
                self.pages[page].visible = False
        return gr.update(visible=True), gr.update(visible=False)
        
    def navigate_select_directory_page(self):
        self.next_page = self.select_directory_button
        return self.navigate("select_directory_page") 
    
    def navigate_label_net_page(self, directory_dropdown, video_dropdown):
        print(f'Navigating to label net page with {directory_dropdown} and {video_dropdown}')
        if not directory_dropdown or not video_dropdown:
            gr.Warning("Please select a valid video file to label.", duration=10)
        else:
            self.pages['label_net_page'].video_path = os.path.join(directory_dropdown, video_dropdown)
            return self.navigate("label_net_page") 
    
    
    def select_directory_button(self):
        directory_dropdown = self.pages.select_directory_page.directory_dropdown
        video_dropdown = self.pages.select_directory_page.video_dropdown
        select_button = gr.Button("Select Video")
        current_page = self.current_page
        
        select_button.click(
                self.navigate_label_net_page, 
                inputs=[directory_dropdown, video_dropdown, current_page.page], 
                outputs=[self.pages.label_net_page.page, current_page.page]
            )

        return select_button