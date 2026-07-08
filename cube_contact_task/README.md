# Cube Goal Task

This directory isolates the cube manipulation task from the older locomotion code.

What it adds:

- A movable MuJoCo cube named `push_cube` in the existing robot scene.
- A lightweight cube with lower friction so the quadruped can move it by contact.
- A visible target marker named `cube_target`.
- Random cube and target respawns around the robot at reset.
- A reward based on reducing cube-to-target distance, with contact and approach shaping.
- Extra observations: cube position, target position, cube-to-target vector, and cube XY velocity.

Run a smoke test:

```bash
python -m cube_contact_task.smoke_cube_push
```

Train the quadruped locally with SBX CrossQ:

```bash
python cube_contact_task/train_quadruped_cube_goal.py \
  --output-dir exp/quadruped_cube_goal_100k \
  --total-steps 100000
```

Train 2M steps on the server:

```bash
python cube_contact_task/train_quadruped_cube_goal.py \
  --output-dir exp/quadruped_cube_goal_2m \
  --total-steps 2000000
```

The branch for this work is:

```bash
cube-contact-reward
```

Push it with:

```bash
git push -u origin cube-contact-reward
```
