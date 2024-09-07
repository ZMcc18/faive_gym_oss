# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# copied from IsaacGymEnvs with slight modifications

import datetime
import isaacgym

import os
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path    # 主要用于将相对路径转换为绝对路径
import gym

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

from isaacgymenvs.utils.utils import set_np_formatting, set_seed    

# register custom tasks for faive_gym here
from isaacgymenvs.tasks import isaacgym_task_map    # 是 Isaac Gym 环境中的一个任务映射，用于加载和管理不同的任务。这种映射机制允许你通过任务名称来检索具体的任务实例
from faive_gym.robot_hand import RobotHand
from faive_gym.tasks.crawl import Crawl
isaacgym_task_map["RobotHand"] = RobotHand
isaacgym_task_map["Crawl"] = Crawl

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
@hydra.main(config_name="config", config_path="./cfg") # 装饰器，配置文件被加载为一个 DictConfig 对象，并传递给 launch_rlg_hydra(cfg) 函数。
def launch_rlg_hydra(cfg: DictConfig):    # 函数 launch_rlg_hydra 的参数 cfg 类型是 DictConfig
    # 用于设置强化学习环境、构建模型和算法，并执行训练任务。
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs

    # 生成时间戳和运行名称，用于在训练过程中的日志记录和结果保存中唯一标识此次运行
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    run_name = f"{cfg.wandb_name}_{time_str}"

    # 确保 cfg.checkpoint 路径是绝对路径，方便模型加载时正确定位。
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # 将 DictConfig 对象转换为 Python 字典，方便进一步操作和传递给外部库。
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # 调整 NumPy 的输出格式以便在终端中输出时更清晰，用来更好地调试或观察张量和数组数据
    set_np_formatting()
    
    # 根据是否使用多 GPU 来设置设备，使用 torchrun 启动多进程训练
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    # 根据配置文件中的种子值进行随机种子设置
    cfg.seed += rank    # 确保在多 GPU 环境下，每个 GPU 使用不同的随机种子
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)

    # 当 cfg.wandb_activate 设置为 True 且当前进程的 rank 为 0 时，使用 wandb（Weights and Biases）进行实验跟踪，方便记录和可视化。
    if cfg.wandb_activate and rank == 0:
        # Make sure to install WandB if you actually use this.
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
            monitor_gym=True,
        )

    def create_env_thunk(**kwargs):
        # 环境创建 (isaacgymenvs.make)
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,    # 指定用来运行物理模拟的设备
            cfg.rl_device,    # 指定用于执行强化学习算法的设备
            cfg.graphics_device_id,    # 指定图形渲染设备的 ID
            cfg.headless,    # 是否启用无头模式。如果设置为 True，那么环境将不进行图形渲染，适合在需要高效训练而不关心视觉输出的情况下使用。无头模式下，环境运行速度通常更快。
            cfg.multi_gpu,    
            cfg.capture_video,    # 视频捕捉功能
            cfg.force_render,    # 是否强制渲染
            cfg,# 包含所有环境和强化学习相关配置的字典，负责传递所有设定。此参数传递给 isaacgymenvs.make 函数，用于进一步初始化环境。
            **kwargs,
        )
        # 开启视频捕获功能
        if cfg.capture_video:
            envs.is_vector_env = True
            # RecordVideo 是 Gym 提供的用于捕捉环境执行过程视频的包装器。它会将 envs 包裹起来，使得每隔一段时间步骤（由 cfg.capture_video_freq 控制）记录一次视频。
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,# 这个触发器函数表示：当当前步数 step 是 capture_video_freq 的倍数时，触发视频记录
                video_length=cfg.capture_video_len,
            )
        return envs

    # register the rl-games adapter to use inside the runner
    # 注册了一个名为 'RLGPU' 的新环境类型，通过匿名函数将 RLGPUEnv 作为该类型的实际实现。
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_thunk,
    })

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        # Runner 是一个强化学习的运行器类，它负责训练和评估强化学习算法。
        # algo_observer 是一个观察者对象，通常用于监控和记录算法的运行情况。
        runner = Runner(algo_observer)
        # 注册算法构建器
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        # 这通常代表一种具体的玩家策略实现，用于与环境进行交互
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict 保存配置文件用的
    experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint' : cfg.checkpoint,
        'sigma' : None
    })

    if cfg.wandb_activate and rank == 0:
        wandb.finish()

if __name__ == "__main__":
    launch_rlg_hydra()
