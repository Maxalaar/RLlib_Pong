import gymnasium
from gymnasium.wrappers import RecordVideo, FrameStack
from ray.tune.registry import register_env


def pong_creator(environment_configuration):
    environment = gymnasium.make(id='ALE/Pong-v5')
    return environment


def record_video_pong_creator(environment_configuration):
    environment = gymnasium.make(id='ALE/Pong-v5', render_mode='rgb_array')
    environment = RecordVideo(environment, video_folder='./ray_videos', episode_trigger=lambda x: x % 1 == 0)
    return environment


register_env(name='my_pong', env_creator=pong_creator)
register_env(name='record_video_pong', env_creator=record_video_pong_creator)
