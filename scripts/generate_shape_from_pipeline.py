import argparse
import ast

from modularlegs.sim.robot_metadesigner import MetaDesignerAsym
from modularlegs.sim.scripts.homemade_robots_asym import MESH_DICT_FINE, ROBOT_CFG_AIR1S


def parse_pipeline(value):
    pipeline = ast.literal_eval(value)
    if not isinstance(pipeline, list) or not all(isinstance(x, int) for x in pipeline):
        raise ValueError("--pipeline must be a Python-style list of integers")
    if len(pipeline) % 4 != 0:
        raise ValueError("Pipeline length must be a multiple of 4")
    return pipeline


def generate_robot(name, pipeline, color="black"):
    designer = MetaDesignerAsym(
        init_pipeline=pipeline,
        robot_cfg=ROBOT_CFG_AIR1S,
        mesh_dict=MESH_DICT_FINE,
    )
    designer.paint(color)
    designer.builder.save(name)
    print(f"saved modularlegs/sim/assets/robots/{name}.xml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Robot XML basename without .xml")
    parser.add_argument("--pipeline", required=True, type=parse_pipeline)
    parser.add_argument("--color", default="black")
    args = parser.parse_args()

    generate_robot(args.name, args.pipeline, args.color)


if __name__ == "__main__":
    main()
