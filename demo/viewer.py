def show_videos_and_scores():
    # Paths to the video files
    final_model_video = "./rendered_videos/car_racing_final_model.avi"
    best_model_video = "./rendered_videos/car_racing_best_model.avi"
    # Scores of the videos, this can be found in tensorboard logs
    final_model_score = 902.5
    best_model_score = 919.9

    return final_model_video, final_model_score, best_model_video, best_model_score
