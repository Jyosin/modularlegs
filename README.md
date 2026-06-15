
# Agile Legged Locomotion in Reconfigurable Modular Robots

This repository contains the codebase for the paper "Agile Legged Locomotion in Reconfigurable Modular Robots". Development is ongoing as we package core functionality into the [`MetaMachine`](https://github.com/Chenaah/Metamachine) library, a platform designed to support multiple modular robot systems. Our plan is to gradually refactor this codebase to depend on `MetaMachine` and use its APIs.

Stay tuned for improvements and expanded functionality. If you run into any issues or have suggestions, we welcome your feedback and contributions!


## Installation

First, clone the repository:
```bash
git clone https://github.com/Chenaah/modularlegs.git
cd modularlegs
```

Create a Conda environment with Python 3.10
```bash
conda create -n modularlegs python=3.10 -y
conda activate modularlegs
```

Install the package:
```bash
pip install -e .
```


## Usage

### Train a specific metamachine

Train a single module:
```bash
python modularlegs/scripts/train_sbx.py sim_train_m3air1s
```

Train a quadruped using curriculum learning:
```bash
python modularlegs/scripts/train_sbx.py curriculum/quadrupedX4air1s
```
Training logs and checkpoints will be saved to the `exp` directory.


### Run a policy on the real metamachines
```bash
python modularlegs/scripts/train_sbx.py real_play_quadrupedX4air1s
```

### Generate custom metamachines
Example of generating a Mujoco XML from a configuration encoding:
```bash
python modularlegs/sim/scripts/homemade_robots_asym.py
```

### Record trained morphology videos with a follow camera
The morphology policies trained under `exp/shape_experiments/<morphology>` can be
recorded without retraining. The recorder generates temporary record configs with
`record.camera_mode: follow` and `trainer.record.camera_mode: follow`, loads each
`rl_model_1000000.zip`, and writes comparison videos to `recordings/`.

Record every completed morphology:
```bash
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
./scripts/record_all_morphologies.sh --model-step 1000000 --record-steps 120
```

Record selected morphologies:
```bash
./scripts/record_all_morphologies.sh --only chain5 arc5 l_shape --model-step 1000000
```

Output files are named with the morphology, for example:
```text
recordings/chain5_follow.mp4
recordings/l_shape_follow.mp4
```

### Run bayesian optimiation 
To use Bayesian optimization, you'll need to install an additional package:
```bash
pip install git+https://github.com/secondmind-labs/trieste.git
```

Next, download the VAE training dataset:
```bash
python data/download.py designs_filtered
```
*Note: The dataset generator script and VAE pretrained checkpoints will be released soon.*

Once the dataset is ready, run the optimization script:
```bash
python modularlegs/scripts/evolve.py evolution_vae_asym_air1s
```


## Citation

If you find this work useful, please cite:

```bibtex
@article{yu2026agile,
  title={Agile legged locomotion in reconfigurable modular robots},
  author={Yu, Chen and Matthews, David and Wang, Jingxian and Gu, Jing and Blackiston, Douglas and Rubenstein, Michael and Kriegman, Sam},
  journal={Proceedings of the National Academy of Sciences},
  volume={123},
  number={10},
  pages={e2519129123},
  year={2026},
  publisher={National Academy of Sciences}
}
```
