import gradio as gr

from viewer import show_videos_and_scores

# Define the Gradio interface
interface = gr.Interface(
    fn=show_videos_and_scores,
    inputs=[],
    outputs=[
        gr.Video(
            label="Final Model Video",
            value="./rendered_videos/car_racing_final_model.avi",
        ),
        gr.Textbox(label="Final Model Score", value="902.5"),
        gr.Video(
            label="Best Model Video",
            value="./rendered_videos/car_racing_best_model.avi",
        ),
        gr.Textbox(label="Best Model Score", value="919.9"),
    ],
    title="Car Racing Videos Demo",
    description="This demo shows the final and best reinforcement learning model videos and scores of the Car Racing environment.",
)

if __name__ == "__main__":
    interface.launch()
