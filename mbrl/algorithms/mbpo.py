# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, Sequence, cast

import gymnasium as gym
import hydra.utils
import numpy as np
import omegaconf
from omegaconf import OmegaConf
import torch

import wandb
from pathlib import Path
import sys

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder

from mbrl.util.logger import WandbLogger

MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]


def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,  # D_env: buffer filled with agent-env rollout data
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,     # D_model: buffer to be filled with agent-model rollout data
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
    ):

    # Sampling a batch of data from D_env
    batch = replay_buffer.sample(batch_size)
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    
    # Model rollouts will begin from the states sampled in the above step
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    
    obs = initial_obs
    
    for i in range(rollout_horizon):

        action = agent.act(obs, sample=sac_samples_action, batched=True)
        
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        
        truncateds = np.zeros_like(pred_dones, dtype=bool)
        
        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
            truncateds[~accum_dones, 0],
        )
        
        obs = pred_next_obs
        
        accum_dones |= pred_dones.squeeze()


def evaluate(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
    video_recorder: VideoRecorder,
) -> float:
    avg_episode_reward = 0.0
    for episode in range(num_episodes):
        obs, _ = env.reset()
        video_recorder.init(enabled=(episode == 0))
        terminated = False
        truncated = False
        episode_reward = 0.0
        while not terminated and not truncated:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def maybe_replace_sac_buffer(
    sac_buffer: Optional[mbrl.util.ReplayBuffer],
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    new_capacity: int,
    seed: int,
) -> mbrl.util.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = sac_buffer.rng
        new_buffer = mbrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if sac_buffer is None:
            return new_buffer
        (
            obs,
            action,
            next_obs,
            reward,
            terminated,
            truncated,
        ) = sac_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, terminated, truncated)
        return new_buffer
    return sac_buffer


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
    ) -> np.float32:

    # Main entry point for MBPO

    # Define the project name as a combination of the script and env names
    project_name = f"{cfg.algorithm.name}_{cfg.overrides.env}"

    # Preparing to log cfg to wandb
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    # Removing action_optimizer as this is not used by MBPO
    cfg_dict.pop("action_optimizer", None)
    clean_cfg = OmegaConf.create(cfg_dict)
    resolved_dict = OmegaConf.to_container(clean_cfg, resolve=True)

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=project_name,
        config=resolved_dict
    )

    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

    work_dir = work_dir or os.getcwd()
    
    # enable_back_compatible to use pytorch_sac agent
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)

    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    
    dtype = np.double if use_double_dtype else np.float32
    
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    
    random_explore = cfg.algorithm.random_initial_explore
    
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=replay_buffer,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )

    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )

    updates_made = 0
    env_steps = 0
    
    # Wrap the dynamics model as a gym-like environment
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, None, generator=torch_generator
    )
    
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
        wandb_logger=wandb_logger,  # Also log to wandb
    )

    best_eval_reward = -np.inf
    epoch = 0
    sac_buffer = None

    while env_steps < cfg.overrides.num_steps:
    
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
    
        # What is happening in these 3 lines?
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
    
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
    
        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )
    
        obs = None
    
        terminated = False
    
        truncated = False
    
        for steps_epoch in range(cfg.overrides.epoch_length):
            
            if steps_epoch == 0 or terminated or truncated:
                steps_epoch = 0
                obs, _ = env.reset()
                terminated = False
                truncated = False

            # --- Doing env step and adding to model dataset ---
            # Creating dataset for training model - at this stage replay buffer is D_env
            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            # --------------- Model Training -----------------
            if (env_steps + 1) % cfg.overrides.freq_train_model == 0:

                # Train model on D_env
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                # This function performs branched rollouts and fills D_model (= sac_buffer)
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                )

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):

                # The agent is sometimes trained on real data (replay_buffer)
                # and sometimes on model generated data (sac_buffer)
                # This seems to be a code-level optimization and is not
                # mentioned in the MBPO paper
                use_real_data = rng.random() < cfg.algorithm.real_data_ratio
                
                which_buffer = replay_buffer if use_real_data else sac_buffer
                
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    which_buffer
                ) < cfg.overrides.sac_batch_size:
                    break  # only update every once in a while

                agent.sac_agent.update_parameters(
                    which_buffer,
                    cfg.overrides.sac_batch_size,
                    updates_made,
                    logger,
                    wandb_logger=wandb_logger,
                    reverse_mask=True,  # What does this do?
                )

                updates_made += 1
                
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                
                avg_reward = evaluate(
                    test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
                )
                
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "episode_reward": avg_reward,
                        "rollout_length": rollout_length,
                    },
                )

                # Log to wandb
                wandb_logger.run.define_metric("eval/*", step_metric="eval/env_step")
                wandb_logger.log_metrics(
                    metrics_dict={
                        "eval/epoch": epoch,
                        "eval/env_step": env_steps,
                        "eval/episode_reward": avg_reward,
                        "eval/rollout_length": rollout_length,
                    },
                )
                
                if avg_reward > best_eval_reward:
                    video_name = f"{epoch}.mp4"
                    video_recorder.save(video_name)
                    best_eval_reward = avg_reward
                    agent.sac_agent.save_checkpoint(
                        ckpt_path=os.path.join(work_dir, "sac.pth")
                    )

                    # Log video to wandb
                    wandb_logger.run.define_metric("video/video_epoch", step_metric="video/epoch")
                    wandb.log({
                        "video/epoch": epoch,
                        "video/video_epoch": wandb.Video(Path(video_recorder.save_dir)/video_name)
                        })

                epoch += 1

            env_steps += 1
            obs = next_obs
            
    return np.float32(best_eval_reward)
