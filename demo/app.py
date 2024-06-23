import gradio as gr
from viewer import show_videos_and_score

# Define the Gradio interface
interface = gr.Interface(
    fn=show_videos_and_score,
    inputs=[],
    outputs=[
        gr.Video(label="Final Model Video"),
        gr.Textbox(label="Final Model Score"),
        gr.Video(label="Best Model Video"),
        gr.Textbox(label="Best Model Score"),
    ],
    title="Car Racing Videos Demo",
    description="This demo shows the final and best model videos of the Car Racing environment.",
)

if __name__ == "__main__":
    interface.launch()
