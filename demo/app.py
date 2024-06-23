import gradio as gr
from video_viewer import show_videos

# Define the Gradio interface
interface = gr.Interface(
    fn=show_videos,
    inputs=[],
    outputs=[gr.Video(label="Final Model Video"), gr.Video(label="Best Model Video")],
    title="Car Racing Videos Demo",
    description="This demo shows the final and best model videos of the Car Racing environment.",
)

if __name__ == "__main__":
    interface.launch()
