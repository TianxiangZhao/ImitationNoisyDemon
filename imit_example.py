import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc,dagger,density,mce_irl,preference_comparisons
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
import os
import ipdb


os.environ["SDL_VIDEODRIVER"] = "dummy"
env = gym.make("CartPole-v1")


def train_expert():
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(100)  # Note: change this to 100000 to trian a decent expert.
    return expert


def sample_expert_transitions():
    expert = train_expert()

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    )
    ipdb.set_trace()
    return rollout.flatten_trajectories(rollouts)


transitions = sample_expert_transitions()

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
)



reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=3, render=False)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=1)

reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=3, render=False)
print(f"Reward after training: {reward}")