# AWS 训练工作总结

日期：2026-06-01

## 目标

这次工作的目标是在 AWS 上搭建可以运行 ModularLegs 的训练环境，并完成 5-actuator quadruped 形态的正式训练和可视化结果导出。

本次训练目标配置：

```text
config/shape_experiments/sim_train_shape_quadruped.yaml
```

该配置的关键参数：

```yaml
agent:
  num_act: 5
sim:
  asset_file: quadrupedX4air1s.xml
```

## AWS 环境

区域：

```text
ap-northeast-1, Tokyo
```

配额情况：

- `Running On-Demand G and VT instances` 配额不足，无法启动 `g5.xlarge` / `g6.xlarge`。
- `Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances` 有可用配额，因此先使用 CPU 实例跑通和训练。

实例选择：

- AMI：Ubuntu Server 24.04 LTS, x86_64
- 实例类型：Standard quota 下的 CPU 实例
- 存储：gp3 EBS，建议 50 GiB
- 登录用户：`ubuntu`

后续如果 GPU quota 获批，建议使用：

```text
g5.xlarge 或 g6.xlarge
```

注意：启动最小的 G 系列 GPU 实例也需要至少 4 个 G/VT vCPU quota。

## 代码同步

本地代码已提交并推送到：

```text
git@github.com:Jyosin/modularlegs.git
```

关键提交：

```text
cf8ef76 Add shape snapshot experiments
59435b3 Use formal snapshot training defaults
```

AWS 上因为没有配置 GitHub SSH key，使用 SSH clone 失败：

```text
git@github.com: Permission denied (publickey)
```

最后使用 HTTPS clone：

```bash
git clone https://github.com/Jyosin/modularlegs.git
cd modularlegs
```

## Python 环境

在 AWS 上创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

因为这次是 CPU 实例，所以没有安装 `jax[cuda12]`，而是安装 CPU 版本 JAX：

```bash
grep -v "jax\\[cuda12\\]" requirements.txt > /tmp/requirements-cpu.txt
pip install -r /tmp/requirements-cpu.txt
pip install "jax[cpu]"
```

运行前设置：

```bash
export JAX_PLATFORMS=cpu
export MUJOCO_GL=osmesa
```

如果其他机器上 `osmesa` 报错，可以尝试：

```bash
export MUJOCO_GL=egl
```

## 脚本修改

修改文件：

```text
scripts/train_shape_snapshots.py
```

原来脚本偏向快速测试，每 200 steps 生成一次 snapshot。现在改成正式训练默认设置：

```python
DEFAULT_TARGET_STEPS = 1_000_000
DEFAULT_SNAPSHOT_INTERVAL = 100_000
DEFAULT_VIDEO_STEPS = 120
```

当前行为：

- 默认训练到 1,000,000 timesteps。
- 每 100,000 timesteps 保存一次 checkpoint。
- 每 100,000 timesteps 生成一次可视化视频。
- 每个视频 rollout 120 steps。

常用命令：

```bash
python scripts/train_shape_snapshots.py --only quadruped
python scripts/train_shape_snapshots.py --only quadruped --target-steps 100000
python scripts/train_shape_snapshots.py --only quadruped --snapshot-interval 50000
python scripts/train_shape_snapshots.py --only quadruped --no-video
```

## 正式训练

AWS 上使用后台方式运行：

```bash
cd ~/modularlegs
source .venv/bin/activate
export JAX_PLATFORMS=cpu
export MUJOCO_GL=osmesa

nohup python -u scripts/train_shape_snapshots.py --only quadruped > quadruped_train.log 2>&1 &
```

查看运行状态：

```bash
tail -f quadruped_train.log
ps -p <PID> -o pid,etime,cmd
top -p <PID>
```

训练已完成，日志中确认保存了：

```text
saved quadruped snapshot at 100000 steps
saved quadruped snapshot at 200000 steps
saved quadruped snapshot at 300000 steps
saved quadruped snapshot at 400000 steps
saved quadruped snapshot at 500000 steps
saved quadruped snapshot at 600000 steps
saved quadruped snapshot at 700000 steps
saved quadruped snapshot at 800000 steps
saved quadruped snapshot at 900000 steps
saved quadruped snapshot at 1000000 steps
```

AWS 上的结果目录：

```text
~/modularlegs/exp/shape_experiments/quadruped
```

模型文件：

```text
rl_model_100000.zip
rl_model_200000.zip
...
rl_model_1000000.zip
rl_model_last.zip
```

可视化目录：

```text
~/modularlegs/exp/shape_experiments/quadruped/visualization
```

## 运行中出现的 Warning

训练中出现过以下 warning，但没有阻止训练完成：

```text
Unable to register cuFFT/cuDNN/cuBLAS factory
TF-TRT Warning: Could not find TensorRT
Failed to log video to wandb: You must call wandb.init() before wandb.log()
Overwriting existing videos at .../visualization folder
```

说明：

- CUDA/TensorRT warning 对这次 CPU 训练无影响。
- W&B warning 只表示没有上传到 W&B，本地视频仍然保存了。
- Overwriting video warning 来自重复使用同一个 visualization 文件夹，不影响模型 checkpoint。

## 下载视频到本地

本地 key 文件：

```text
wang_macbook.pem
```

该文件已加入 `.gitignore`，不要提交到 GitHub。

设置 key 权限：

```bash
chmod 400 "/Users/wangruqin/Desktop/2026春/RA/modularlegs/wang_macbook.pem"
```

下载视频到本地 repo 根目录下的 `quadruped_videos/`：

```bash
mkdir -p "/Users/wangruqin/Desktop/2026春/RA/modularlegs/quadruped_videos"

scp -i "/Users/wangruqin/Desktop/2026春/RA/modularlegs/wang_macbook.pem" \
  'ubuntu@<EC2_PUBLIC_IP>:~/modularlegs/exp/shape_experiments/quadruped/visualization/*.mp4' \
  "/Users/wangruqin/Desktop/2026春/RA/modularlegs/quadruped_videos/"
```

注意：远程路径需要用引号包起来，否则本地 `zsh` 会尝试展开 `*.mp4`，导致：

```text
zsh: no matches found
```

下载完整 visualization 文件夹：

```bash
scp -i "/Users/wangruqin/Desktop/2026春/RA/modularlegs/wang_macbook.pem" -r \
  ubuntu@<EC2_PUBLIC_IP>:~/modularlegs/exp/shape_experiments/quadruped/visualization \
  "/Users/wangruqin/Desktop/2026春/RA/modularlegs/quadruped_visualization"
```

打开本地视频目录：

```bash
open "/Users/wangruqin/Desktop/2026春/RA/modularlegs/quadruped_videos"
```

## 从新 AWS CPU 实例复现

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip libgl1 libgl1-mesa-dev libosmesa6-dev libegl1-mesa-dev ffmpeg

git clone https://github.com/Jyosin/modularlegs.git
cd modularlegs

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
grep -v "jax\\[cuda12\\]" requirements.txt > /tmp/requirements-cpu.txt
pip install -r /tmp/requirements-cpu.txt
pip install "jax[cpu]"

export JAX_PLATFORMS=cpu
export MUJOCO_GL=osmesa
nohup python -u scripts/train_shape_snapshots.py --only quadruped > quadruped_train.log 2>&1 &
tail -f quadruped_train.log
```

## 后续建议

- 检查每个 100k checkpoint 的视频，确认学习过程是否符合预期。
- 保留 `rl_model_1000000.zip`、`rl_model_last.zip` 和 visualization 目录。
- 如果后续 G 系列 GPU quota 获批，在 `g5.xlarge` / `g6.xlarge` 上重新跑同一脚本。
- 如果需要在线实验记录，可以显式初始化 W&B；如果只需要本地视频，可以忽略当前 W&B warning。
