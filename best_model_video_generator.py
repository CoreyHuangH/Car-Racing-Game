import os
import gymnasium as gym
import cv2
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from envs.discrete_car_racing import DiscreteCarRacing

def generate_best_model_video():
    # Create environment
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    # Convert environment to DiscreteCarRacing
    env = DiscreteCarRacing(env)

    # Check environment
    check_env(env)

    # Monitor environment
    env = Monitor(env)
    # Wrap environment
    env = DummyVecEnv([lambda: env])

    model = DQN.load("./models/best_model", env=env)

    # Render final model to video
    obs = env.reset()
    os.makedirs("./rendered_videos", exist_ok=True)
    video_writer = cv2.VideoWriter(
        "./rendered_videos/car_racing_best_model.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        30,
        (env.render().shape[1], env.render().shape[0]),
    )

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        video_writer.write(frame)
        if dones:
            obs = env.reset()
    
    video_writer.release()
    env.close()

if __name__ == "__main__":
    generate_best_model_video()