import numpy as np

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, TASK_MAPPING, register_problem

_TabletopBase = TASK_MAPPING["libero_tabletop_manipulation"]

# 로봇 배치: None이면 기존 동작(책상 바깥, x=±(0.16+L/2), z=0)
# 값이 주어지면 robot0=-X, robot1=+X 위치, 책상 위(z=table_top)에 올림
ROBOT_X_OFFSET = None

# None이면 로봇 클래스 기본 init_qpos 사용
HOME_QPOS = None


@register_problem
class LIBERO_Dual_Tabletop_Manipulation(_TabletopBase):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "main_table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.table_full_size = kwargs["table_full_size"]
        else:
            self.table_full_size = (1.0, 1.2, 0.05)
        self.table_offset = (0, 0, 0.90)
        self.z_offset = 0.01 - self.table_full_size[2]

        kwargs.update(
            {"robots": [f"OnTheGround{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "table"})

        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/libero_tabletop_base_style.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "light-gray",
                        "wall_style": "light-gray-plaster",
                    }
                }
            )

        BDDLBaseDomain.__init__(self, bddl_file_name, *args, **kwargs)

    def _check_robot_configuration(self, robots):
        pass  # 1개 또는 2개 모두 허용

    def _load_model(self):
        super()._load_model()

        if HOME_QPOS is not None:
            home_qpos = np.array(HOME_QPOS, dtype=float)
            for robot in self.robots:
                robot.init_qpos = home_qpos.copy()

        # 책상 윗면 z (table_offset z; default 0.8)
        try:
            table_top_z = float(self.mujoco_arena.table_offset[2])
        except Exception:
            table_top_z = 0.8

        if ROBOT_X_OFFSET is not None:
            # 두 로봇 모두 책상 위에 배치 (robot0=-X, robot1=+X)
            self.robots[0].robot_model.set_base_xpos(
                np.array([-ROBOT_X_OFFSET, 0.0, table_top_z])
            )
            self.robots[0].robot_model.set_base_ori(np.array([0, 0, 0]))
            if len(self.robots) == 2:
                self.robots[1].robot_model.set_base_xpos(
                    np.array([+ROBOT_X_OFFSET, 0.0, table_top_z])
                )
                self.robots[1].robot_model.set_base_ori(np.array([0, 0, np.pi]))
        else:
            # 기존 동작: robot1만 반대편 바깥쪽으로
            if len(self.robots) == 2:
                table_length = self.table_full_size[0]
                xpos = np.array([0.16 + table_length / 2, 0, 0])
                rot = np.array([0, 0, np.pi])
                self.robots[1].robot_model.set_base_xpos(xpos)
                self.robots[1].robot_model.set_base_ori(rot)
