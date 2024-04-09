from typing import Any

from gymnasium.core import ActType
from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer
from mujoco import mjtGeom

from ..task import Task
from mujoco_env.common.defs.types import Pos3D
from .scoring import multi_object_position_epsilon_success_score


class COMRearrangementTask(Task):

    def reset(self, obj_poses: dict[str, dict[str, Pos3D]], time_limit: int, obs_poses=None, epsilon=0.1):
        self.obj_poses = obj_poses
        self.time_limit = time_limit
        self.epsilon = epsilon

        self.task_obj_names = list(self.obj_poses.keys())
        self.task_objs = [self.sim.get_entity(name) for name in self.task_obj_names]
        self.obj_goals = {name: self.obj_poses[name]['goal_com'] for name in self.task_obj_names}

        self.step_count = 0

        for obj_pose, obj in zip(self.obj_poses.values(), self.task_objs):
            obj.configure_joints(position=obj_pose['start_pose'])

        if obs_poses is not None:
            obs_objs = [self.sim.get_entity(name) for name in obs_poses.keys()]
            for obs_pose, obs in zip(obs_poses.values(), obs_objs):
                obs.configure_joints(position=obs_pose['start_pose'])

    def begin_frame(self, action: ActType) -> None:
        pass

    def end_frame(self, action: ActType) -> None:
        self.step_count += 1

    def score(self) -> float:
        # score will be 1 if all objects are within epsilon of their goal COM
        score = multi_object_position_epsilon_success_score(
            x1=[obj.center_of_mass for obj in self.task_objs],
            x2=list(self.obj_goals.values()),
            epsilon=self.epsilon
        )

        return float(score) * self.is_done()  # rewarded only when done

    def is_done(self) -> bool:
        return self.step_count >= self.time_limit  # done if time limit is reached

    def get_info(self) -> dict[str, Any]:
        return dict(
            time_limit=self.time_limit,
            step_count=self.step_count,
            task_obj_goal_com=self.obj_goals,
        )

    def update_render(self, viewer: WindowViewer):
        for obj_name, pos in self.obj_poses.items():
            viewer.add_marker(
                pos=pos['goal_com'],
                size=[self.epsilon] * 3,
                rgba=[0, .9, 0, 0.3],
                type=mjtGeom.mjGEOM_SPHERE,
                label=obj_name
            )
