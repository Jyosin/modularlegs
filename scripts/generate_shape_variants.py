try:
    from generate_shape_from_pipeline import generate_robot
except ImportError:
    from scripts.generate_shape_from_pipeline import generate_robot


SHAPE_VARIANTS = {
    "chain5_air1s": {
        "label": "straight 5-module chain",
        "pipeline": [0, 17, 0, 0, 1, 17, 0, 0, 2, 17, 0, 0, 3, 17, 0, 0],
    },
    "arc5_air1s": {
        "label": "curved serial chain",
        "pipeline": [0, 17, 0, 1, 1, 17, 0, 1, 2, 17, 0, 1, 3, 17, 0, 1],
    },
    "zigzag5_air1s": {
        "label": "alternating zigzag chain",
        "pipeline": [0, 17, 0, 0, 1, 17, 0, 1, 2, 17, 0, 0, 3, 17, 0, 1],
    },
    "branch_y_air1s": {
        "label": "Y branch from the second module",
        "pipeline": [0, 17, 0, 0, 1, 17, 0, 0, 1, 7, 0, 0, 1, 9, 0, 0],
    },
    "comb5_air1s": {
        "label": "two-module spine with side branches",
        "pipeline": [0, 17, 0, 0, 1, 17, 0, 0, 0, 7, 0, 0, 1, 9, 0, 1],
    },
    "cross4_air1s": {
        "label": "four limbs from the root module",
        "pipeline": [0, 1, 0, 0, 0, 3, 0, 0, 0, 13, 0, 0, 0, 15, 0, 0],
    },
    "offset_cross_air1s": {
        "label": "rotated four-limb root shape",
        "pipeline": [0, 2, 0, 0, 0, 4, 0, 1, 0, 14, 0, 0, 0, 16, 0, 1],
    },
    "t_shape_air1s": {
        "label": "T branch on a short spine",
        "pipeline": [0, 17, 0, 0, 1, 17, 0, 0, 1, 7, 0, 0, 1, 8, 0, 0],
    },
    "t_wide_air1s": {
        "label": "wide T with branches on both ball ports",
        "pipeline": [0, 17, 0, 0, 1, 17, 0, 0, 1, 7, 0, 0, 1, 8, 0, 1],
    },
    "t_offset_air1s": {
        "label": "offset T with the crossbar near the root",
        "pipeline": [0, 17, 0, 0, 1, 17, 0, 0, 0, 7, 0, 0, 0, 8, 0, 1],
    },
    "t_long_stem_air1s": {
        "label": "long-stem T with fork at the tip",
        "pipeline": [0, 17, 0, 0, 1, 17, 0, 0, 2, 17, 0, 0, 2, 7, 0, 0],
    },
    "l_shape_air1s": {
        "label": "simple L shape",
        "pipeline": [0, 17, 0, 0, 1, 17, 0, 0, 2, 7, 0, 0, 3, 17, 0, 0],
    },
    "l_hook_air1s": {
        "label": "hook-like L shape",
        "pipeline": [0, 17, 0, 0, 1, 17, 0, 0, 2, 7, 0, 1, 3, 7, 0, 0],
    },
    "l_stair_air1s": {
        "label": "stair-step L shape",
        "pipeline": [0, 17, 0, 0, 1, 7, 0, 0, 2, 17, 0, 0, 3, 7, 0, 0],
    },
    "front_fork_air1s": {
        "label": "fork at the tip of a three-module spine",
        "pipeline": [0, 17, 0, 0, 1, 17, 0, 0, 2, 7, 0, 0, 2, 9, 0, 0],
    },
    "double_tail_air1s": {
        "label": "two tails from opposite ends of the root",
        "pipeline": [0, 17, 0, 0, 0, 0, 0, 0, 1, 17, 0, 0, 2, 17, 0, 0],
    },
}


if __name__ == "__main__":
    for name, spec in SHAPE_VARIANTS.items():
        generate_robot(name, spec["pipeline"])
