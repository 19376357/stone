import os
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from InukshukEnv_v0 import InukshukEnv_v0
from InukshukEnv_v1 import InukshukEnv_v1
from InukshukEnv_v2 import InukshukEnv_v2
from InukshukEnv_v3 import InukshukEnv_v3
import stable_baselines3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import imageio
from callback import SaveOnBestTrainingRewardCallback, HParamCallback, VideoRecorderCallback, \
    TensorboardCallback, SummaryWriterCallback
from stable_baselines3.common.callbacks import \
    CheckpointCallback, CallbackList, ProgressBarCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":

    train = True
    # True False

    env = InukshukEnv_v3()

    # Create log dir
    log_dir = "logger/tmp/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir, info_keywords=('similarity', 'reward_placed', 'reward_y', 'reward_holes'))

    if train:

        # create model
        model = stable_baselines3.PPO("MultiInputPolicy", env, learning_rate=5e-4, gamma=0.95, verbose=2, ent_coef=0.05, tensorboard_log="Logger/tmp/InukshukEnv_tensorboard")

        # load pretrained weight
        #model.set_parameters('logger/ppo_model_100000_steps.zip', exact_match=True)

        # Create the callback: check every 1000 steps
        #savebest_callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
        tensorboard_callback = TensorboardCallback(check_freq=100, log_dir=log_dir, verbose=0)

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=15, verbose=1)
        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="logger/",
            name_prefix="ppo_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        video_recorder = VideoRecorderCallback(env, render_freq=5000)
        # Train the agent
        timesteps = 1e6
        model.learn(total_timesteps=int(timesteps), tb_log_name="first_run", callback=CallbackList(
            [checkpoint_callback, tensorboard_callback, ProgressBarCallback(), HParamCallback(),
             video_recorder]))

        # plot
        plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO Inukshuk")
        plt.savefig('logger/tmp/ppo.png')
        plt.show()


    else:

        model = stable_baselines3.PPO("MultiInputPolicy", env, verbose=2)
        model.set_parameters('logger/ppo_model_500000_steps.zip', exact_match=True)

        env = model.get_env()
        #obs = env.reset()



    # save gif
    images = []
    obs = env.reset()
    img = env.render(mode="rgb_array")
    for i in range(200):
        images.append(img)
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        img = model.env.render(mode="rgb_array")
    imageio.mimsave("logger/inuksuk.gif",
                    [np.array(img) for i, img in enumerate(images)], fps=5)
    model.save('logger/weight')

    env.close()

    '''
    obs = env.reset()
    p.setRealTimeSimulation(1)
    while True:
        # action = env.action_space.sample()
        action = np.random.uniform(-1, 1, size=(7,))

        obs, reward, done, _ = env.step(action)
        if done:
            break

        #print(f"state : {obs}, reward : {reward}")'''
