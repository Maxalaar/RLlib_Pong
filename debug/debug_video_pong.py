import gymnasium
from gymnasium.wrappers import RecordVideo, FrameStack

from environment.environment_creator import record_video_pong_creator

if __name__ == '__main__':
    environment = record_video_pong_creator({})
    environment.reset()
    for i in range(1_000_000):
        observation, reward, terminated, truncated, info = environment.step(environment.action_space.sample())
        if terminated:
            environment.reset()
        # print(observation.shape)
    environment.close()
    environment.close_video_recorder()
