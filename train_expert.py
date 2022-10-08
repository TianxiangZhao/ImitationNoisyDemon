import gym
import numpy as np
import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
# import mujoco_py
import torch
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
import os
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.env_util import make_vec_env
from gym_minigrid.wrappers import FlatObsWrapper, ImgObsWrapper, RGBImgObsWrapper, FullyObsWrapper
from envs.obs_wrapper import ImgFlatObsWrapper, IndexObsWrapper
import ipdb
from envs.new_fourrooms import *
# --------------
# args
# --------------
os.environ["SDL_VIDEODRIVER"] = "dummy"
#env_name = 'MiniGrid-FourRooms-v1'
env_name = 'MiniGrid-DoorKey-5x5-v0'
#env_name = 'MiniGrid-LavaGapS5-v0'
#env_name = 'MiniGrid-MultiRoom-N2-S4-v0'
#env_name = 'MiniGrid-DistShift1-v0'
print(env_name)
n_env = 8
seed = 42
#FIX_TASK = True
FIX_TASK = False
goal = [13,16]
train_step = 1000000
lr = 5e-4
bs = 64
# n_step = 1024 #256 for FourRooms
n_step = 256

# ------------------
# env creation
# ------------------
if FIX_TASK:
    env_kwargs = {'goal_pos': goal}
    save_path = 'data/{}_goal{}_Index'.format(env_name, goal)
else:
    env_kwargs = None
    save_path = 'data/{}_Index'.format(env_name)
os.makedirs(save_path, exist_ok=True)

#env = gym.make(env_name, **env_kwargs)
#env = Monitor(env)
#env = FullyObsWrapper(env)
#env = FlatObsWrapper(env)
#env = make_vec_env(env_name,n_envs=n_env,wrapper_class=IndexObsWrapper, seed=seed, env_kwargs=env_kwargs)
env = make_vec_env(env_name,n_envs=n_env,wrapper_class=ImgFlatObsWrapper, seed=seed, env_kwargs=env_kwargs)
env.max_steps = n_step
for i in range(len(env.envs)):#for dummy vec env
    env.envs[i].max_steps = n_step

obs = env.reset()


# -------------------
# train policy model
# -------------------

model = PPO(MlpPolicy, env, verbose=0, seed=seed, n_steps=n_step, batch_size=bs, learning_rate=lr,tensorboard_log=save_path+'_log/')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

eval_callback = EvalCallback(env,
                             best_model_save_path=save_path,
                             log_path=save_path,
                             eval_freq=train_step // (40*n_env),
                             n_eval_episodes=100,
                             deterministic=True,
                             render=False)
model.learn(total_timesteps=train_step, callback=eval_callback)


mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"final mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

model.save(save_path+'/model_'+str(seed)+'.pth')
model.save(save_path+'/model_'+str(seed)+'batchsize{}nstep{}'.format(bs, n_step)+'.pth')

model.logger.record("batch size", bs)
model.logger.record("n_step", n_step)

# -----------------------------------------
# evaluation
# -----------------------------------------
env = gym.make(env_name)
env = Monitor(env)
env = ImgFlatObsWrapper(env)
video_dir = "./videos/"
os.makedirs(video_dir, exist_ok=True)
obs = env.reset()
imgs = []
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    # action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    img = env.render(mode='rgb_array')
    imgs.append(img)
    if done:
        obs = env.reset()
    # if i//15 ==0:
      # fig.clf()
      # plt.imshow(img)
      # plt.show()

if FIX_TASK:
    imageio.mimsave(video_dir+'{}_goal{}.gif'.format(env_name,goal), [np.array(img) for i, img in enumerate(imgs) if i%2 == 0], fps=29)
else:
    imageio.mimsave(video_dir+'{}_ori.gif'.format(env_name), [np.array(img) for i, img in enumerate(imgs) if i%2 == 0], fps=29)

print('finished')
