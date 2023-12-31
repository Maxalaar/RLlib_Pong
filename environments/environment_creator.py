import gymnasium
from gymnasium.wrappers import RecordVideo
from ray.rllib.env.multi_agent_env import make_multi_agent, MultiAgentEnvWrapper
from ray.rllib.env.wrappers.atari_wrappers import WarpFrame, FrameStack
from ray.tune.registry import register_env


def pong_creator(environment_configuration):
    environment = gymnasium.make(id='ALE/Pong-v5', frameskip=1, full_action_space=False, repeat_action_probability=0.0)
    environment = WarpFrame(environment, dim=42)
    environment = FrameStack(environment, 4)
    return environment


def record_video_pong_creator(environment_configuration):
    environment = gymnasium.make(id='ALE/Pong-v5', render_mode='rgb_array', frameskip=1, full_action_space=False, repeat_action_probability=0.0)
    environment = RecordVideo(environment, video_folder='./ray_videos', episode_trigger=lambda x: True)
    return environment


def mutli_agent_pong(environment_configuration):
    environment = make_multi_agent(pong_creator)
    environment = environment({"num_agents": 1})
    return environment


register_env(name='my_pong', env_creator=pong_creator)
register_env(name='record_video_pong', env_creator=record_video_pong_creator)
register_env(name='mutli_agent_pong', env_creator=mutli_agent_pong)