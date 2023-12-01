import gymnasium
from gymnasium.wrappers import RecordVideo

if __name__ == '__main__':
    environment = gymnasium.make(id='ALE/Pong-v5', render_mode='rgb_array')  # , render_mode='human'
    environment = RecordVideo(environment, video_folder='./ray_videos')
    environment.reset()
    for i in range(1_000_000):
        observation, reward, terminated, truncated, info = environment.step(environment.action_space.sample())
    environment.close()
    environment.close_video_recorder()
