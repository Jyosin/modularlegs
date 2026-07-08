import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cube_contact_task.train_quadruped_cube_goal import main


if __name__ == "__main__":
    main()
