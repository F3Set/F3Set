import os
import gradio as gr
from interfaces.select_directory_page import SelectDirectoryPage
from interfaces.label_net_page import LabelNetPage
from interfaces.label_page import LabelPage

def main():
    with gr.Blocks(css="assets/styles.css") as app:
        
        # Initialize pages
        select_directory_page = SelectDirectoryPage()
        label_net_page = LabelNetPage(visible=False)
        label_page = LabelPage(visible=False)
        
        # Initialize navigation buttons for select_directory_page
        label_net_page_button = select_directory_page.build_label_net_page_button(label_net_page)
        
        # Initialize navigation buttons for label_net_page
        label_net_page.setup_prev_page_button(select_directory_page)
        label_net_page.setup_next_page_button(label_page)
        
        # Initialize navigation buttons for label_page
        label_page.setup_prev_page_button(label_net_page)
        
        return app
        
if __name__ == "__main__":
    app = main()
    app.launch()