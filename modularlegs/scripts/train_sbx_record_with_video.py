from collections import defaultdict
import copy
import os
os.environ["MUJOCO_GL"] = "egl"
import pdb
import shutil
import time
import yaml
import numpy as np
import argparse
import gymnasium as gym
from omegaconf import OmegaConf
import sbx
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

# 画像出力用に追加
import glob
import json
import imageio.v2 as imageio


# from modularlegs import LEG_ROOT_DIR
try:
    from modularlegs import LEG_ROOT_DIR
except ImportError:
    LEG_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from modularlegs.envs.gym.rendering import RecordVideo
from modularlegs.envs.env_sim import ZeroSim
from modularlegs.envs.env_real import Real
from modularlegs.envs.wrappers import VecReal
from modularlegs.utils.train import EpisodicRewardCallback, ProgressBarCallbackName, multiplex_obs, save_rollout, load_model
from modularlegs.utils.files import get_cfg_name, get_cfg_path, get_curriculum_cfg_paths, update_cfg, load_cfg, get_latest_model
from modularlegs.utils.logger import get_running_header, plot_learning_curve
from modularlegs.utils.model import is_headless
from modularlegs.utils.others import is_list_like


def _to_serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    return x


def patch_pickle_compat_for_numpy_jax():
    """
    NumPy 2.x / 新旧 JAX 環境で保存された SB3/SBX モデルを、
    現在の NumPy 1.26.x / jaxlib 環境で読むための互換 patch。
    """

    import sys
    import types
    import importlib

    # ============================================================
    # 1. numpy._core -> numpy.core alias
    # ============================================================
    try:
        import numpy
        import numpy.core

        if "numpy._core" not in sys.modules:
            m = types.ModuleType("numpy._core")
            m.__dict__.update(numpy.core.__dict__)
            m.__path__ = []  # package 風にする
            sys.modules["numpy._core"] = m

        for name in [
            "numeric",
            "multiarray",
            "umath",
            "fromnumeric",
            "shape_base",
            "function_base",
            "arrayprint",
            "records",
            "memmap",
            "numerictypes",
            "overrides",
            "_exceptions",
            "_multiarray_umath",
        ]:
            try:
                mod = importlib.import_module(f"numpy.core.{name}")
                sys.modules.setdefault(f"numpy._core.{name}", mod)
            except ModuleNotFoundError:
                pass

    except Exception as e:
        print("WARNING: numpy compatibility patch failed:", repr(e))

    # ============================================================
    # 2. numpy.random BitGenerator pickle compatibility
    # ============================================================
    try:
        import numpy
        import numpy.random._pickle as nr_pickle
        from numpy.random import Generator, RandomState

        bitgen_map = {}

        for cls_name in ["MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"]:
            try:
                bitgen_map[cls_name] = getattr(numpy.random, cls_name)
            except AttributeError:
                pass

        if hasattr(nr_pickle, "BitGenerators"):
            nr_pickle.BitGenerators.update(bitgen_map)

        def normalize_bitgen_name(bit_generator_name):
            if isinstance(bit_generator_name, type):
                return bit_generator_name.__name__

            s = str(bit_generator_name)

            if "PCG64DXSM" in s:
                return "PCG64DXSM"
            if "PCG64" in s:
                return "PCG64"
            if "MT19937" in s:
                return "MT19937"
            if "Philox" in s:
                return "Philox"
            if "SFC64" in s:
                return "SFC64"

            if isinstance(bit_generator_name, str):
                return bit_generator_name.split(".")[-1]

            return s.split(".")[-1].replace("'>", "")

        def patched_bit_generator_ctor(bit_generator_name="MT19937"):
            name = normalize_bitgen_name(bit_generator_name)

            if name in bitgen_map:
                # 重要: dict ではなく BitGenerator の instance を返す
                return bitgen_map[name]()

            if hasattr(nr_pickle, "BitGenerators") and name in nr_pickle.BitGenerators:
                return nr_pickle.BitGenerators[name]()

            raise ValueError(
                f"{bit_generator_name!r} is not a known BitGenerator module. "
                f"normalized name={name!r}, known={list(bitgen_map.keys())}"
            )

        def patched_generator_ctor(bit_generator_name="MT19937", bit_generator_ctor=None):
            bit_generator = patched_bit_generator_ctor(bit_generator_name)
            return Generator(bit_generator)

        def patched_randomstate_ctor(bit_generator_name="MT19937", bit_generator_ctor=None):
            return RandomState()

        nr_pickle.__bit_generator_ctor = patched_bit_generator_ctor
        nr_pickle.__generator_ctor = patched_generator_ctor
        nr_pickle.__randomstate_ctor = patched_randomstate_ctor

    except Exception as e:
        print("WARNING: numpy.random compatibility patch failed:", repr(e))

    # ============================================================
    # 3. jaxlib._jax / jaxlib._jax.pytree compatibility
    # ============================================================
    try:
        import jaxlib

        # すでに存在するなら何もしない
        try:
            import jaxlib._jax
            try:
                import jaxlib._jax.pytree
            except ModuleNotFoundError:
                pass
        except ModuleNotFoundError:
            pass

        import jaxlib.xla_extension as xe

        # --------------------------------------------------------
        # jaxlib._jax を package 風 module として作る
        # --------------------------------------------------------
        if "jaxlib._jax" not in sys.modules:
            jax_mod = types.ModuleType("jaxlib._jax")
            jax_mod.__dict__.update(xe.__dict__)
            jax_mod.__package__ = "jaxlib._jax"
            jax_mod.__path__ = []  # これが重要: package として扱わせる
            sys.modules["jaxlib._jax"] = jax_mod
            setattr(jaxlib, "_jax", jax_mod)
        else:
            jax_mod = sys.modules["jaxlib._jax"]
            if not hasattr(jax_mod, "__path__"):
                jax_mod.__path__ = []

        # --------------------------------------------------------
        # jaxlib._jax.pytree を用意する
        # --------------------------------------------------------
        if "jaxlib._jax.pytree" not in sys.modules:
            pytree_mod = None

            # 候補1: jaxlib.xla_extension.pytree が import できる場合
            try:
                pytree_mod = importlib.import_module("jaxlib.xla_extension.pytree")
            except Exception:
                pytree_mod = None

            # 候補2: xe.pytree 属性がある場合
            if pytree_mod is None and hasattr(xe, "pytree"):
                candidate = getattr(xe, "pytree")
                if isinstance(candidate, types.ModuleType):
                    pytree_mod = candidate
                else:
                    pytree_mod = types.ModuleType("jaxlib._jax.pytree")
                    pytree_mod.__dict__.update(getattr(candidate, "__dict__", {}))

            # 候補3: fallback module を作る
            if pytree_mod is None:
                pytree_mod = types.ModuleType("jaxlib._jax.pytree")

                # よく pickle に出る PyTree 関連 object を xe から移す
                for attr in [
                    "PyTreeDef",
                    "pytree",
                    "flatten",
                    "unflatten",
                ]:
                    if hasattr(xe, attr):
                        setattr(pytree_mod, attr, getattr(xe, attr))

                # jax.tree_util 側も補助的に入れる
                try:
                    import jax.tree_util as tree_util

                    for attr in [
                        "PyTreeDef",
                        "tree_flatten",
                        "tree_unflatten",
                        "tree_leaves",
                        "tree_structure",
                    ]:
                        if hasattr(tree_util, attr):
                            setattr(pytree_mod, attr, getattr(tree_util, attr))
                except Exception:
                    pass

            pytree_mod.__name__ = "jaxlib._jax.pytree"
            pytree_mod.__package__ = "jaxlib._jax"

            sys.modules["jaxlib._jax.pytree"] = pytree_mod
            setattr(jax_mod, "pytree", pytree_mod)

        print("patched: jaxlib._jax and jaxlib._jax.pytree compatibility enabled")

    except Exception as e:
        print("WARNING: jaxlib compatibility patch failed:", repr(e))

    # ============================================================
    # 4. optax.tree compatibility
    # ============================================================
    try:
        import optax

        try:
            import optax.tree
        except ModuleNotFoundError:
            import types

            tree_mod = types.ModuleType("optax.tree")
            tree_mod.__package__ = "optax"

            # 古い Optax では tree_utils に tree_* 関数がある
            try:
                import optax.tree_utils as tree_utils

                # tree_utils 内の関数をそのまま登録
                for attr in dir(tree_utils):
                    if not attr.startswith("_"):
                        setattr(tree_mod, attr, getattr(tree_utils, attr))

                # 新しい optax.tree 風の短い名前も作る
                alias_map = {
                    "add": "tree_add",
                    "sub": "tree_sub",
                    "mul": "tree_mul",
                    "div": "tree_div",
                    "scale": "tree_scale",
                    "zeros_like": "tree_zeros_like",
                    "ones_like": "tree_ones_like",
                    "full_like": "tree_full_like",
                    "norm": "tree_norm",
                    "sum": "tree_sum",
                    "max": "tree_max",
                    "min": "tree_min",
                    "vdot": "tree_vdot",
                    "where": "tree_where",
                    "clip": "tree_clip",
                    "allclose": "tree_allclose",
                }

                for new_name, old_name in alias_map.items():
                    if hasattr(tree_utils, old_name):
                        setattr(tree_mod, new_name, getattr(tree_utils, old_name))

            except Exception as e:
                print("WARNING: optax.tree_utils alias failed:", repr(e))

            sys.modules["optax.tree"] = tree_mod
            setattr(optax, "tree", tree_mod)

            print("patched: optax.tree -> optax.tree_utils compatibility")

    except Exception as e:
        print("WARNING: optax compatibility patch failed:", repr(e))
    # ============================================================
    # 5. jax._src.tree_util.none_leaf_registry compatibility
    # ============================================================
    try:
        import jax._src.tree_util as jtu

        if not hasattr(jtu, "none_leaf_registry"):
            if hasattr(jtu, "default_registry"):
                jtu.none_leaf_registry = jtu.default_registry
                print("patched: jax._src.tree_util.none_leaf_registry -> default_registry")
            else:
                jtu.none_leaf_registry = None
                print("patched: jax._src.tree_util.none_leaf_registry -> None")

    except Exception as e:
        print("WARNING: jax tree_util compatibility patch failed:", repr(e))
    # ============================================================
    # 6. jax._src.named_sharding compatibility
    # ============================================================
    try:
        import sys
        import types
        import jax

        # すでに存在するなら何もしない
        try:
            import jax._src.named_sharding
        except ModuleNotFoundError:
            named_sharding_mod = types.ModuleType("jax._src.named_sharding")
            named_sharding_mod.__package__ = "jax._src"

            # 現在の JAX では public API 側にあることが多い
            try:
                from jax.sharding import NamedSharding, PartitionSpec, Mesh

                named_sharding_mod.NamedSharding = NamedSharding
                named_sharding_mod.PartitionSpec = PartitionSpec
                named_sharding_mod.Mesh = Mesh

            except Exception:
                # fallback: jax._src.sharding_impl 側を探す
                try:
                    import jax._src.sharding_impl as sharding_impl

                    for attr in [
                        "NamedSharding",
                        "PartitionSpec",
                        "Mesh",
                        "PmapSharding",
                        "SingleDeviceSharding",
                    ]:
                        if hasattr(sharding_impl, attr):
                            setattr(named_sharding_mod, attr, getattr(sharding_impl, attr))

                except Exception as e:
                    print("WARNING: jax._src.sharding_impl fallback failed:", repr(e))

            # 念のため他の sharding 関連属性も public API から移す
            try:
                import jax.sharding as jax_sharding

                for attr in dir(jax_sharding):
                    if not attr.startswith("_") and not hasattr(named_sharding_mod, attr):
                        setattr(named_sharding_mod, attr, getattr(jax_sharding, attr))

            except Exception:
                pass

            sys.modules["jax._src.named_sharding"] = named_sharding_mod

            # jax._src に属性としても生やす
            try:
                import jax._src as jax_src
                setattr(jax_src, "named_sharding", named_sharding_mod)
            except Exception:
                pass

            print("patched: jax._src.named_sharding compatibility enabled")

    except Exception as e:
        print("WARNING: jax._src.named_sharding compatibility patch failed:", repr(e))


def _episode_max_steps(conf):
    return None if conf.agent.done_version is None else 1000


def _maybe_time_limit(env, conf):
    max_episode_steps = _episode_max_steps(conf)
    if max_episode_steps is None:
        return env
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)


class Trainer:

    def __init__(self, conf_list):
        self.conf_list = conf_list
        self.is_env_setup = False
        self.is_device_set = False
        self.curriculum = False

        if "curriculum" in conf_list[0]:
            self.curriculum = True
            assert len(conf_list) == 1, "Curriculum learning only supports one configuration"
            self.conf_list = get_curriculum_cfg_paths(conf_list[0])
            curriculum_steps = [load_cfg(conf_name, alg="sbx").trainer.curriculum_step for conf_name in self.conf_list]
            # Sort the configurations based on the curriculum steps
            self.conf_list = [a for _, a in sorted(zip(curriculum_steps, self.conf_list))]

        master_conf = self.conf_list[0]

        def update_robot_log_dir(conf):
            # Update the robot data directory ONCE
            if conf.robot.mode == "real":
                if conf.logging.robot_data_dir is not None:
                    if conf.logging.robot_data_dir == "auto":
                        conf.logging.robot_data_dir = conf.logging.data_dir # One path for all the configurations
                    os.makedirs(conf.logging.robot_data_dir, exist_ok=True)
                # Also update the logging notes
                conf.trainer.notes += f"\n{get_running_header()}"

            return conf
       
        self._update_conf(master_conf, conf_update_func=update_robot_log_dir)
        self.master_conf = copy.deepcopy(self.conf)

        # 強制的にrecordモードで実行
        self.conf.trainer.mode = "record"
        #self.conf.trainer.record.record_steps = 1000
        #self.conf.trainer.record.num_envs = 1
        
        
        if self.conf.robot.mode == "real":
            self._save_conf(self.conf.logging.robot_data_dir)

        # Set up wandb
        run = wandb.init(
            project="OctopusLite",
            name=f"{self.conf.robot.mode}-{self.conf.agent.obs_version}-{self.conf.agent.reward_version}",
            config=OmegaConf.to_container(self.conf),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            # monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            mode="online" if self.conf.trainer.wandb_on else "disabled"
        )

        if self.conf.trainer.joystick and self.conf.trainer.mode == "play":
            raise NotImplementedError("Joystick is not supported yet")



    def _setup_env(self):
        # Setting up the environment
        # The environment only set up once
        if self.conf.robot.mode == "sim":
            if self.conf.sim.render and is_headless():
                self.conf.sim.render = False
                print("Running in headless mode; render is turned off!")
            self.unwarpped_env = ZeroSim(self.conf)
            self.env = _maybe_time_limit(self.unwarpped_env, self.conf)
            
            if self.conf.trainer.num_envs >1:
                # TODO: this case in real world 
                env_funs = [lambda: Monitor(_maybe_time_limit(ZeroSim(self.conf), self.conf))]*self.conf.trainer.num_envs
                self.env = DummyVecEnv(env_funs)

            elif self.conf.agent.num_envs > 1:
                tenv = _maybe_time_limit(self.unwarpped_env, self.conf)
                trigger = lambda t: t % 199 == 0
                self.env = RecordVideo(tenv, 
                                video_folder=self.conf.logging.data_dir, 
                                episode_trigger=trigger, 
                                fps=1/self.conf.robot.dt,
                                disable_logger=True)
                self.env = VecReal(self.env, max_episode_steps=_episode_max_steps(self.conf))

            elif not self.conf.sim.render and self.conf.trainer.mode in ["train", "play"]:
                trigger = lambda t: t % 199 == 0
                self.env = RecordVideo(self.env, 
                                video_folder=self.conf.logging.data_dir, 
                                episode_trigger=trigger, 
                                fps=1/self.conf.robot.dt,
                                disable_logger=True)
        elif self.conf.robot.mode == "real":
            self.unwarpped_env = Real(self.conf)
            if self.conf.agent.num_envs == 1:
                self.env = _maybe_time_limit(self.unwarpped_env, self.conf)
            else:
                self.env = VecReal(self.unwarpped_env, max_episode_steps=_episode_max_steps(self.conf))
        else:
            raise ValueError("Invalid robot mode: ", self.conf.robot.mode)
        
    def _get_cfg_path_flexible(self, conf_name):
        """
        conf_name が既存ファイルならその絶対パスを返す。
        そうでなければ従来の get_cfg_path(conf_name) を使う。
        """
        if isinstance(conf_name, str) and os.path.isfile(conf_name):
            return os.path.abspath(conf_name)
        return os.path.abspath(get_cfg_path(conf_name))

    def _load_cfg_flexible(self, conf_name):
        """
        conf_name が:
          - 実在する設定ファイルパスなら OmegaConf.load()
          - それ以外なら既存の load_cfg()
        """
        if isinstance(conf_name, str) and os.path.isfile(conf_name):
            cfg_path = os.path.abspath(conf_name)
            conf = OmegaConf.load(cfg_path)
        else:
            cfg_path = os.path.abspath(get_cfg_path(conf_name))
            conf = load_cfg(conf_name, alg="sbx")

        return conf, cfg_path

    def _resolve_record_model_paths(self, conf_name, conf):
        """
        record モードで使うモデル(zip)の一覧を解決する。

        優先順位:
          1) trainer.load_run が文字列 -> その1本
          2) trainer.load_run がリスト -> その全て
          3) trainer.load_run が None -> config ファイルのあるディレクトリの *.zip 全て
        """
        load_run = conf.trainer.load_run

        if is_list_like(load_run):
            model_paths = [os.path.abspath(p) for p in load_run if p is not None]

        elif load_run is not None:
            model_paths = [os.path.abspath(load_run)]

        else:
            cfg_path = self._get_cfg_path_flexible(conf_name)
            cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
            model_paths = sorted(glob.glob(os.path.join(cfg_dir, "*.zip")))

            if len(model_paths) == 0:
                raise FileNotFoundError(
                    f"No .zip model files were found in config directory: {cfg_dir}\n"
                    f"Please set trainer.load_run explicitly or place model zip files there."
                )

        return model_paths

    def _make_model_output_dir(self, model_path):
        """
        読み込む zip ファイル名(拡張子なし)で出力フォルダを作る。
        例:
          /path/to/rl_model_100000_steps.zip
            -> /path/to/rl_model_100000_steps/
        """
        model_path = os.path.abspath(model_path)
        parent_dir = os.path.dirname(model_path)
        model_filename = os.path.basename(model_path)
        model_stem = os.path.splitext(model_filename)[0]

        output_dir = os.path.join(parent_dir, model_stem)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "_source_model.txt"), "w") as f:
            f.write(model_path + "\n")

        return output_dir

    def _update_conf(self, conf_name, reset_env=False, conf_update_func=None):
        self.conf, self.conf_path = self._load_cfg_flexible(conf_name)

        if conf_update_func is not None:
            self.conf = conf_update_func(self.conf)

        self.conf_name = conf_name

        if hasattr(self, "master_conf"):
            self.conf.interface.module_ids = self.master_conf.interface.module_ids
            self.conf.interface.torso_module_id = self.master_conf.interface.torso_module_id

        if not self.is_device_set:
            # Set the device
            device = self.conf.trainer.device
            if "cuda" in device:
                os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
                os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ["JAX_PLATFORMS"] = "cpu"
            self.is_device_set = True

        if hasattr(sbx, self.conf.trainer.algorithm):
            self.Alg = getattr(sbx, self.conf.trainer.algorithm)
        else:
            raise ValueError(f"Algorithm {self.conf.trainer.algorithm} not found in sbx")

        # Only set up the environment once
        if (not self.is_env_setup) or reset_env:
            self._setup_env()
            self.is_env_setup = True

        # Setting up the model
        self.unwarpped_env.update_config(self.conf)

        alg_kwargs = (
            OmegaConf.to_container(self.conf.trainer.algorithm_params)
            if self.conf.trainer.algorithm_params is not None
            else {}
        )

        # record モードで load_run=None の場合は、
        # _record() の中で zip を列挙して個別ロードするので、
        # ここでは eager に model をロードしない
        if self.conf.trainer.mode == "record" and self.conf.trainer.load_run is None:
            self.model = None
            self.models = []
            self.vec_env = None
            return

        if not is_list_like(self.conf.trainer.load_run):
            self.model = load_model(
                self.conf.trainer.load_run,
                self.env,
                self.Alg,
                alg_kwargs=alg_kwargs
            )
            if self.conf.trainer.load_replay_buffer is not None:
                self.model.load_replay_buffer(self.conf.trainer.load_replay_buffer)
            self.models = [self.model]
            self.vec_env = self.model.get_env()
        else:
            self.models = [
                load_model(load_run, self.env, self.Alg, alg_kwargs=alg_kwargs)
                for load_run in self.conf.trainer.load_run
            ]
            self.vec_env = self.models[0].get_env()
            self.model = self.models[0]


    def _save_conf(self, log_dir):
        shutil.copy(self.conf_path, log_dir)
        with open(os.path.join(log_dir, "running_config.yaml"), "w") as file:
            yaml.dump(OmegaConf.to_container(self.conf, resolve=True), file, default_flow_style=False)
        with open(os.path.join(log_dir, "note.txt"), 'w') as f:
            f.write(self.conf.trainer.notes)
        asset_files = [self.conf.sim.asset_file] if not is_list_like(self.conf.sim.asset_file) else self.conf.sim.asset_file
        asset_log_dir = os.path.join(log_dir, "assets")
        os.makedirs(asset_log_dir, exist_ok=True)
        for asset_file in asset_files:
            xml_file = os.path.join(LEG_ROOT_DIR, "modularlegs", "sim", "assets", "robots", asset_file)
            shutil.copy(xml_file, asset_log_dir)

    def _train(self):
        last_log_dir = None
        for i, conf_name in enumerate(self.conf_list):

            def update_log_dir(conf):
                conf.trainer.load_run = last_log_dir
                return conf

            if not i == 0:
                self._update_conf(conf_name, reset_env=True, conf_update_func=update_log_dir if self.curriculum else None)

            # Setting up the logger
            log_dir =self.conf.logging.data_dir
            logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
            checkpoint_callback = CheckpointCallback(
                save_freq=100000 if self.conf.robot.mode == "sim" else int(2000*self.conf.agent.num_envs),
                save_path=log_dir,
                name_prefix="rl_model",
                save_replay_buffer=self.conf.robot.mode == "real",
                save_vecnormalize=True,
                )
            wandb_callback = WandbCallback(log="all")
            self._save_conf(log_dir)

            self.model.set_logger(logger)
            rich_callback = ProgressBarCallbackName(get_cfg_name(conf_name))
            callbacks = [checkpoint_callback, wandb_callback, rich_callback] if self.conf.robot.mode == "sim" else [checkpoint_callback, wandb_callback]
            if self.conf.agent.num_envs > 1:
                callbacks.append(EpisodicRewardCallback())

            # Training the model
            self.model.learn(total_timesteps=self.conf.trainer.total_steps, 
                        callback=callbacks) # TODO: progress bar for real world training
            
            self.model.save(os.path.join(log_dir, "rl_model_last.zip"))

            plot_learning_curve(os.path.join(log_dir, "progress.csv"), os.path.join(log_dir, "curve.png"))
            
            last_log_dir = get_latest_model(log_dir)


    def _play(self):
        self.recovering = False
        self.multiplexing = self.conf.trainer.multiplex
        self.unwarpped_env.commands = [0,1,0]
        obs = self.vec_env.reset()
        step_count = 0
        while True:
            t0 = time.time()

            if self.multiplexing:
                if self.conf.trainer.multiplex_type == "4+1":
                    obs_tuple = multiplex_obs(obs, "4+1")
                    actions = []
                    for obs, model in zip(obs_tuple, self.models):
                        act, _states = model.predict(obs, deterministic=True)
                        actions.append(act)
                    action = np.concatenate((actions[0][:,:3], actions[1],actions[0][:,3:4]), axis=1)

                elif self.conf.trainer.multiplex_type == "3+1+1":
                    obs_tuple = multiplex_obs(obs, "3+1+1")
                    actions = []
                    for obs, model in zip(obs_tuple, self.models):
                        act, _states = model.predict(obs, deterministic=True)
                        actions.append(act)
                    action = np.concatenate((actions[0][:,:3], actions[1],actions[2]), axis=1)

            else:

                action, _states = self.model.predict(obs, deterministic=True)


            obs, reward, done, info = self.vec_env.step(action)

            # Switching the policy according to priority
            # print("Current policy: ", self.conf_name)
            policy_switch = info[0]["policy_switch"]
            upsidedown = info[0]["upsidedown"]
            chopped = info[0]["chopped"]

            joystick_data = self.joystick_server.pull_data() if self.conf.trainer.joystick and self.conf.trainer.mode == "play" else None
            joystick_command = self.joystick_compiler.get_command(joystick_data) if joystick_data is not None else None
            # print("Joystick command: ", joystick_command)
            if upsidedown and self.conf.trainer.auto_recovery:
                # Switch to the recovery policy if the robot is upside down
                conf_name = self.conf.trainer.recovery_config
                self._update_conf(conf_name)
                self.recovering = True
                obs = self.vec_env.reset()
                self.start_recovery = time.time()

            elif self.recovering and time.time() - self.start_recovery > self.conf.trainer.recovery_time and not upsidedown:
                # Switch back to the original policy after 3 seconds of recovery
                conf_name = self.conf_list[0]
                self._update_conf(conf_name)
                obs = self.vec_env.reset()
                self.recovering = False

            elif self.conf.trainer.monitored_module in chopped and self.conf.trainer.auto_multiplex:
                print("chopped: ", chopped)
                conf_name = self.conf.trainer.multiplex_config
                self._update_conf(conf_name)
                obs = self.vec_env.reset()
                self.multiplexing = True

            elif policy_switch is not None:
                # Switch to the policy specified by keyboad input
                print(f"Policy switch: {policy_switch}")
                assert isinstance(policy_switch, int), "Policy switch must be an integer"
                self.vec_env.close()
                conf_name = self.conf_list[policy_switch]
                self._update_conf(conf_name)
                obs = self.vec_env.reset()

            elif joystick_command is not None:
                print(f"Joystick command: {joystick_command}")
                # Switching the policy according to joystick input
                if joystick_command == "right_bumper":
                    # jumpCCW
                    conf_name = self.conf.trainer.candidate_configs[0]
                elif joystick_command == "left_bumper":
                    # jump
                    conf_name = self.conf.trainer.candidate_configs[1]
                elif joystick_command == "right_trigger" or joystick_command == "left_trigger":
                    # Walking policy
                    conf_name = self.conf.trainer.candidate_configs[2]
                # elif :
                #     conf_name = "real_play_quadrupedX4air1s_back"
                elif joystick_command == "neutral":
                    conf_name = self.conf_list[0]
                else:
                    conf_name = self.conf_list[0]

                self._update_conf(conf_name)
                obs = self.vec_env.reset()
                self.unwarpped_env.commands[0] = 1
                
                
            if self.conf.robot.mode == "sim" and self.conf.sim.render:
                time.sleep(max(0, t0 + self.conf.robot.dt - time.time()))

            step_count += 1

    def _record(self):

        def _find_first_existing_body(env, candidates):
            env = env.unwrapped
            for name in candidates:
                try:
                    env.data.body(name).xpos
                    return name
                except Exception:
                    pass
            return None
        
        
        def _get_body_pos(env, body_name=None):
            """
            body_name が指定されていればその body の位置を返す。
            指定なし・失敗時は qpos[:3] を使う。
            """
            env = env.unwrapped
        
            if body_name is not None:
                try:
                    return env.data.body(body_name).xpos.copy()
                except Exception:
                    pass
        
            # よくありそうな胴体名を順に試す
            auto_body = _find_first_existing_body(
                env,
                ["base", "torso", "trunk", "body", "chassis", "root"]
            )
            if auto_body is not None:
                return env.data.body(auto_body).xpos.copy()
        
            # free joint の root 位置として qpos[:3] を使うフォールバック
            try:
                return env.data.qpos[:3].copy()
            except Exception:
                return None
        
        
        def _update_follow_camera_without_xml(
            env,
            body_name=None,
            distance=3.0,
            elevation=-20.0,
            azimuth=90.0,
            z_offset=0.3,
        ):
            """
            XMLを変更せず、Python側でfree cameraの注視点をロボット位置へ追従させる。
            viewer.cam または mujoco_renderer.viewer.cam がある実装を想定。
            """
            env = env.unwrapped
            pos = _get_body_pos(env, body_name=body_name)
            if pos is None:
                return False
        
            lookat = pos.copy()
            lookat[2] += z_offset
        
            cam = None
        
            # 実装1: env.viewer.cam
            if hasattr(env, "viewer") and hasattr(env.viewer, "cam"):
                cam = env.viewer.cam
        
            # 実装2: Gymnasium系 MujocoEnv の mujoco_renderer.viewer.cam
            elif hasattr(env, "mujoco_renderer"):
                renderer = env.mujoco_renderer
                if hasattr(renderer, "viewer") and hasattr(renderer.viewer, "cam"):
                    cam = renderer.viewer.cam
        
            # 実装3: 独自 renderer.cam
            elif hasattr(env, "renderer") and hasattr(env.renderer, "cam"):
                cam = env.renderer.cam
        
            if cam is None:
                return False
        
            cam.lookat[:] = lookat
            cam.distance = distance
            cam.elevation = elevation
            cam.azimuth = azimuth
        
            return True
        
        for conf_name in self.conf_list:
            conf, cfg_path = self._load_cfg_flexible(conf_name)

            top_record_cfg = conf.get("record", {})
            record_cfg = OmegaConf.merge(top_record_cfg, conf.trainer.get("record", {}))
            agent_cfg = conf.get("agent", {})

            record_obs_version = record_cfg.get(
                "obs_version",
                agent_cfg.get("obs_version", None),
            )

            num_envs = 1 #int(record_cfg.get("num_envs", 1))
            batch_steps = int(record_cfg.get("record_steps", 1000) // num_envs)
            print(batch_steps)

            # video config
            record_video = bool(record_cfg.get("video", True))
            video_env_index = int(record_cfg.get("video_env_index", 0))
            video_every_n_steps = int(record_cfg.get("video_every_n_steps", 1))
            video_fps = float(
                record_cfg.get("fps", 1 / (conf.robot.dt * video_every_n_steps))
            )
            video_name = record_cfg.get("video_name", "episode_000.mp4")
            save_obs_jsonl = bool(record_cfg.get("save_obs_jsonl", True))

            if video_env_index < 0 or video_env_index >= num_envs:
                raise ValueError(
                    f"video_env_index={video_env_index} is out of range for num_envs={num_envs}"
                )

            # load_run が null なら config ディレクトリの zip を全部拾う
            model_paths = self._resolve_record_model_paths(conf_name, conf)

            print(f"[record] config={conf_name}")
            print(f"[record] cfg_path={cfg_path}")
            print(f"[record] resolved {len(model_paths)} model(s):")
            for p in model_paths:
                print(f"  - {p}")

            for model_path in model_paths:
                print("=" * 80)
                print(f"[record] Loading model: {model_path}")

                # モデル別出力フォルダ
                save_dir = self._make_model_output_dir(model_path)
                print(f"[record] Output dir: {save_dir}")

                # render を明示的に有効化（headless EGL 前提）
                if hasattr(conf, "sim"):
                    conf.sim.render = False

                def make_env():
                    return _maybe_time_limit(ZeroSim(conf), conf)

                env_for_model = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=DummyVecEnv)

                alg_kwargs = (
                    OmegaConf.to_container(conf.trainer.algorithm_params)
                    if conf.trainer.algorithm_params is not None
                    else {}
                )

                patch_pickle_compat_for_numpy_jax()
                
                model = load_model(model_path, env_for_model, self.Alg, alg_kwargs=alg_kwargs)
                vec_env = model.get_env()

                obs = vec_env.reset()
                constructed_obs = (
                    [
                        vec_env.envs[i].unwrapped.brain._construct_obs(record_obs_version)
                        for i in range(len(vec_env.envs))
                    ]
                    if record_obs_version is not None
                    else obs
                )

                rollout = defaultdict(list)

                # 動画・ログ出力先
                video_path = os.path.join(save_dir, video_name)
                obslog_path = os.path.join(save_dir, "episode_000.obs.jsonl")
                video_ref_path = os.path.join(save_dir, "rollout_video_refs.json")

                video_writer = None
                if record_video:
                    video_writer = imageio.get_writer(video_path, fps=video_fps)

                obslog_fp = open(obslog_path, "w", encoding="utf-8") if save_obs_jsonl else None

                t0 = time.time()

                from rich.progress import Progress
                progress = Progress()
                progress.start()
                task = progress.add_task(
                    f"[red]Recording {os.path.basename(model_path)}...",
                    total=batch_steps
                )

                step_idx = 0
                frame_idx = 0

                try:
                    while True:
                        action, _states = model.predict(obs, deterministic=True)

                        # step 前の obs を保存
                        rollout["observations"].append(constructed_obs)

                        act_recorded = (
                            action
                            if not record_cfg.get("normalize_default_pos", False)
                            else action + np.array(conf.agent.default_dof_pos)
                        )
                        rollout["actions"].append(act_recorded)

                        obs, reward, done, info = vec_env.step(action)

                        constructed_obs = (
                            [
                                vec_env.envs[i].unwrapped.brain._construct_obs(record_obs_version)
                                for i in range(len(vec_env.envs))
                            ]
                            if record_obs_version is not None
                            else obs
                        )

                        rollout["rewards"].append(reward)
                        rollout["dones"].append(done)

                        # frame_idx を rollout に入れる
                        # shape: (num_envs,)
                        # 動画対象 env のみ 0,1,2,... / それ以外は -1
                        current_frame_idx = -1
                        frame_idx_arr = np.full((num_envs,), -1, dtype=np.int32)

                        if record_video and (step_idx % video_every_n_steps == 0):
                            target_env = vec_env.envs[video_env_index]
                        
                            camera_mode = record_cfg.get("camera_mode", "fixed")
                        
                            if camera_mode == "follow":
                                _update_follow_camera_without_xml(
                                    target_env,
                                    body_name=record_cfg.get("camera_follow_body", None),
                                    distance=float(record_cfg.get("camera_distance", 3.0)),
                                    elevation=float(record_cfg.get("camera_elevation", -20.0)),
                                    azimuth=float(record_cfg.get("camera_azimuth", 90.0)),
                                    z_offset=float(record_cfg.get("camera_z_offset", 0.3)),
                                )
                        
                            frame = target_env.render()
                        
                            if frame is not None:
                                frame = np.asarray(frame, dtype=np.uint8)
                                video_writer.append_data(frame)
                                current_frame_idx = frame_idx
                                frame_idx_arr[video_env_index] = current_frame_idx
                                frame_idx += 1

                        rollout["frame_idx"].append(frame_idx_arr)

                        # JSONL sidecar
                        if obslog_fp is not None:
                            row = {
                                "step_idx": step_idx,
                                "frame_idx": int(current_frame_idx),
                                "video_env_index": int(video_env_index),
                                "obs": _to_serializable(
                                    constructed_obs[video_env_index]
                                    if isinstance(constructed_obs, (list, tuple))
                                    else constructed_obs
                                ),
                                "action": _to_serializable(
                                    act_recorded[video_env_index]
                                    if hasattr(act_recorded, "__len__")
                                    else act_recorded
                                ),
                                "reward": _to_serializable(
                                    reward[video_env_index]
                                    if hasattr(reward, "__len__")
                                    else reward
                                ),
                                "done": _to_serializable(
                                    done[video_env_index]
                                    if hasattr(done, "__len__")
                                    else done
                                ),
                            }
                            obslog_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

                        progress.update(task, advance=1)

                        if time.time() - t0 > 60 * 30:
                            print(f"[record] Intermediate save: {len(rollout['observations'])} steps -> {save_dir}")
                            save_rollout(rollout, save_dir)

                            with open(video_ref_path, "w", encoding="utf-8") as f:
                                json.dump(
                                    {
                                        "source_model": model_path,
                                        "video_path": os.path.relpath(video_path, save_dir),
                                        "video_env_index": video_env_index,
                                        "fps": video_fps,
                                        "video_every_n_steps": video_every_n_steps,
                                        "frame_index_semantics": "rollout['frame_idx'][t][i] == frame index in video, or -1 if no frame was written for env i at step t",
                                    },
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                            t0 = time.time()

                        if len(rollout["observations"]) >= batch_steps:
                            print(f"[record] Finished ({len(rollout['observations'])} steps). Saving to {save_dir} ...")
                            save_rollout(rollout, save_dir)

                            with open(video_ref_path, "w", encoding="utf-8") as f:
                                json.dump(
                                    {
                                        "source_model": model_path,
                                        "video_path": os.path.relpath(video_path, save_dir),
                                        "video_env_index": video_env_index,
                                        "fps": video_fps,
                                        "video_every_n_steps": video_every_n_steps,
                                        "frame_index_semantics": "rollout['frame_idx'][t][i] == frame index in video, or -1 if no frame was written for env i at step t",
                                    },
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )

                            print(f"[record] Done: {save_dir}")
                            break

                        step_idx += 1

                finally:
                    if obslog_fp is not None:
                        obslog_fp.close()
                    if video_writer is not None:
                        video_writer.close()
                    progress.stop()
                    vec_env.close()


    def run(self):
            
        mode = self.conf.trainer.mode

        if mode == "train":
            # Training the model
            self._train()
            
        elif mode == "play":
            # Testing the model
            self._play()

        elif mode == "record":
            # Recording the trajectories
            num_workers = self.conf.trainer.record.num_workers
            if num_workers > 1:
                raise NotImplementedError("Recording with multiple workers is not supported yet")
            else:
                self._record()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', nargs='+', default=['sim_train_m3air1s'])
    args = parser.parse_args()

    trainer = Trainer(args.cfg)
    trainer.run()
    
