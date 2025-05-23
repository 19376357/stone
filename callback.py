import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from typing import Any, Dict
import gym as gym
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.logger import TensorBoardOutputFormat

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True

class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0.0,
            "train/clip_fraction": 0.0,
            "train/clip_range": 0.0,
            "train/entropy_loss": 0.0,
            "train/explained_variance": 0.0,
            "train/learning_rate": 0.0,
            "train/loss": 0.0,
            "train/n_updates": 0.0,
            "train/policy_gradient_loss": 0.0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

class SummaryWriterCallback(BaseCallback):

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        logs = load_results("logger/tmp/")
        similarity = logs['similarity'].values
        holes = logs['reward_holes'].values
        reward_y = logs['reward_y'].values
        x, y = ts2xy(logs, "timesteps")
        if len(x) > 0:
            self.tb_formatter.writer.add_scalar("reward/reward", y[-1], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("reward/similarity", similarity[-1], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("reward/holes", holes[-1], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("reward/reward_y", reward_y[-1], self.num_timesteps)
            self.tb_formatter.writer.flush()
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.best_mean_similarity = -np.inf
        self.best_mean_holes = -np.inf
        self.best_mean_reward_y = -np.inf


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            logs = load_results(self.log_dir)
            similarity = logs['similarity'].values
            holes = logs['reward_holes'].values
            reward_y = logs['reward_y'].values
            x, y = ts2xy(logs, "timesteps")
            # Retrieve training reward
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                mean_similarity = np.mean(similarity[-100:])
                mean_holes = np.mean(holes[-100:])
                mean_reward_y = np.mean(reward_y[-100:])
                self.logger.record("reward/mean_reward", mean_reward)
                self.logger.record("reward/mean_similarity", mean_similarity)
                self.logger.record("reward/mean_holes", mean_holes)
                self.logger.record("reward/mean_reward_y", mean_reward_y)

                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                    print(
                        f"Best mean similarity: {self.best_mean_similarity:.2f} - Last mean similarity per episode: {mean_similarity:.2f}")
                    print(
                        f"Best mean holes: {self.best_mean_holes:.2f} - Last mean holes per episode: {mean_holes:.2f}")
                if mean_similarity > self.best_mean_similarity:
                    self.best_mean_similarity = mean_similarity
                if mean_holes > self.best_mean_holes:
                    self.best_mean_holes = mean_holes
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.best_mean_similarity = -np.inf
        self.best_mean_holes = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            logs = load_results(self.log_dir)
            similarity = logs['similarity'].values
            holes = logs['reward_holes'].values
            x, y = ts2xy(logs, "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                mean_similarity = np.mean(similarity[-100:])
                mean_holes = np.mean(holes[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                    print(
                        f"Best mean similarity: {self.best_mean_similarity:.2f} - Last mean similarity per episode: {mean_similarity:.2f}")
                    print(
                        f"Best mean holes: {self.best_mean_holes:.2f} - Last mean holes per episode: {mean_holes:.2f}")
                if mean_similarity > self.best_mean_similarity:
                    self.best_mean_similarity = mean_similarity
                if mean_holes > self.best_mean_holes:
                    self.best_mean_holes = mean_holes
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True