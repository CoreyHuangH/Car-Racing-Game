import gradio as gr


def show_videos():
    # Paths to the video files
    final_model_video = "./rendered_videos/car_racing_final_model.avi"
    best_model_video = "./rendered_videos/car_racing_best_model.avi"

    return final_model_video, best_model_video


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
