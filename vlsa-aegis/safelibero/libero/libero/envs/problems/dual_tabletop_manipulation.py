import xml.etree.ElementTree as ET
import numpy as np

from libero.libero.envs.bddl_base_domain import TASK_MAPPING, register_problem

_TabletopBase = TASK_MAPPING["libero_tabletop_manipulation"]

# preview_dual.py 등에서 env 생성 전에 덮어쓰기 가능
SIDE_DEPTH    = None   # x 방향 깊이, None이면 table_length/2 자동
SIDE_WIDTH    = 0.5    # y 방향 너비 (m)
SIDE_Y_OFFSET = None   # main_table 중심에서 y 거리, None이면 table_width/2 + side_width/2
SIDE_X_OFFSET = 0.0    # 로봇 기준 x 추가 오프셋 (m)


@register_problem
class LIBERO_Dual_Tabletop_Manipulation(_TabletopBase):

    def _check_robot_configuration(self, robots):
        pass  # 1개 또는 2개 모두 허용

    def _load_model(self):
        super()._load_model()

        if len(self.robots) == 2:
            table_length = self.table_full_size[0]
            xpos = np.array([0.16 + table_length / 2, 0, 0])
            rot = np.array([0, 0, np.pi])
            self.robots[1].robot_model.set_base_xpos(xpos)
            self.robots[1].robot_model.set_base_ori(rot)

    def _setup_camera(self, mujoco_arena):
        super()._setup_camera(mujoco_arena)
        self._add_side_tables_to_arena(mujoco_arena)

    def _add_side_tables_to_arena(self, mujoco_arena):
        table_length = self.table_full_size[0]
        table_width  = self.table_full_size[1]
        table_height = self.table_full_size[2]

        # arena에서 table body z 위치 가져오기
        table_z = 0.0
        for body in mujoco_arena.worldbody.iter('body'):
            if 'table' in body.get('name', '').lower():
                pos = body.get('pos', '0 0 0').split()
                table_z = float(pos[2])
                break

        robot0_x = -(0.16 + table_length / 2)
        robot1_x =  (0.16 + table_length / 2)

        import libero.libero.envs.problems.dual_tabletop_manipulation as _mod
        side_depth = _mod.SIDE_DEPTH if _mod.SIDE_DEPTH is not None else table_length / 2
        side_width = _mod.SIDE_WIDTH
        y_offset   = _mod.SIDE_Y_OFFSET if _mod.SIDE_Y_OFFSET is not None else table_width / 2 + side_width / 2
        x_offset   = _mod.SIDE_X_OFFSET

        configs = [
            (robot0_x + x_offset, -y_offset, 'side_table_0l'),
            (robot0_x + x_offset, +y_offset, 'side_table_0r'),
            (robot1_x + x_offset, -y_offset, 'side_table_1l'),
            (robot1_x + x_offset, +y_offset, 'side_table_1r'),
        ]

        for x, y, name in configs:
            body = ET.Element('body', name=name, pos=f'{x} {y} {table_z}')
            # collision geom (group 0)
            ET.SubElement(body, 'geom',
                name=f'{name}_col',
                type='box',
                size=f'{side_depth/2} {side_width/2} {table_height/2}',
                rgba='0.76 0.61 0.44 1',
                group='0',
                contype='1',
                conaffinity='1',
            )
            # visual geom (group 1)
            ET.SubElement(body, 'geom',
                name=f'{name}_vis',
                type='box',
                size=f'{side_depth/2} {side_width/2} {table_height/2}',
                material='table_texture',
                group='1',
                contype='0',
                conaffinity='0',
            )
            mujoco_arena.worldbody.append(body)
