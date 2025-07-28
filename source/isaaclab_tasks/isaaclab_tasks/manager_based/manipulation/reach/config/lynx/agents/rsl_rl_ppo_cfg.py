# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


##
# Policy configuration
##

@configclass
class LynxReachRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"


@configclass
class LynxReachRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    value_loss_coef = 1.0
    clip_param = 0.2
    entropy_coef = 0.001
    num_learning_epochs = 4
    num_mini_batches = 4
    learning_rate = 1e-3
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class LynxReachRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    # policy
    policy: LynxReachRslRlPpoActorCriticCfg = LynxReachRslRlPpoActorCriticCfg()
    # algorithm
    algorithm: LynxReachRslRlPpoAlgorithmCfg = LynxReachRslRlPpoAlgorithmCfg()

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 1000
    experiment_name = "LynxReach"
    empirical_normalization = True