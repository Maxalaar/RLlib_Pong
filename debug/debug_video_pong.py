import gymnasium
from gymnasium.wrappers import RecordVideo, FrameStack

if __name__ == '__main__':
    environment = gymnasium.make(id='ALE/Pong-v5')  # , render_mode='human' , render_mode='rgb_array'
    # environment = RecordVideo(environment, video_folder='./ray_videos')
    # environment = FrameStack(environment, 4, True)
    environment.reset()
    for i in range(1_000_000):
        observation, reward, terminated, truncated, info = environment.step(environment.action_space.sample())
        print(observation.shape)
    environment.close()
    environment.close_video_recorder()
