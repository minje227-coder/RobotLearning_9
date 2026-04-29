import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OpposingMountedPanda(ManipulatorModel):
    """맞은편에 배치되는 Panda — table 반대쪽 끝에 180도 회전해서 마운트"""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/panda/robot.xml"), idn=idn)
        self.set_joint_attribute(
            attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
        )

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_panda"

    @property
    def init_qpos(self):
        return np.array(
            [0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4]
        )

    @property
    def base_xpos_offset(self):
        return {
            "table": lambda table_length: (0.16 + table_length / 2, 0, 0),  # 반대편
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
