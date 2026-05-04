# 변경사항 요약 (vs main branch)

## 개요
원본 `auto_scene.py` (단일 robot이 milk를 trash로 옮기는 시연)을 확장하여
- **New 안 적용** (한쪽에 물체들 + 반대편 박스 1개)
- **장애물 충돌 시나리오** (병에 부딪힘)
- **정적 robot1 팔 충돌 시나리오**
- **양쪽 동시 동작 (bimanual) 충돌 시나리오**

까지 단계별로 만들었음. 모든 변경은 `Data generation/auto_scene.py` 한 파일에 집중.

---

## 변경된 파일

| 파일 | 상태 |
|---|---|
| `Data generation/auto_scene.py` | **수정** (총 ~237줄 diff) |
| `Data generation/AUTO_SCENE.mp4` | 변경 (정상 trajectory 재생성) |
| `Data generation/AUTO_SCENE_OBSTACLE.mp4` | **신규** (병 충돌 시나리오) |
| `Data generation/AUTO_SCENE_ARM_COLLIDE.mp4` | **신규** (정적 robot1 팔 충돌) |
| `Data generation/AUTO_SCENE_BIMANUAL.mp4` | **신규** (양쪽 동시 동작 충돌) |
| `CHANGES.md` | **신규** (이 파일) |

다른 파일 (`vlsa-aegis/`, BDDL 정의, libero 코드 등) **변경 없음**.

---

## auto_scene.py 변경 상세

### 1. 객체 배치 좌표 (New 안 + bimanual 대칭)

| 상수 | 원본 | 최종 | 의도 |
|---|---|---|---|
| `MILK_CENTER` | `(-0.35, -0.35)` | `(-0.30, -0.20)` | robot0 왼쪽 영역 |
| `MILK_RADIUS` | `0.14` | `0.05` | 샘플링 범위 좁힘 |
| `DISTRACTOR_MILK_CENTER` | `(0.40, 0.50)` | `(0.30, -0.20)` | x=0 거울 대칭 (robot1 측) |
| `DISTRACTOR_MILK_RADIUS` | `0.14` | `0.05` | |
| `TRASH_CAN_CENTER` | `(0.00, 0.35)` | `(0.00, 0.30)` | x=0 위, 두 robot 수렴점 |
| `TRASH_CAN_2_CENTER` | (자동 mirror) | `(0.40, 0.45)` | 외곽으로 밀어 무시 |

### 2. 신규 상수 추가

```python
# 장애물 시나리오용
ENABLE_OBSTACLE = False                    # 충돌 시연 토글
OBSTACLE_CENTER = (-0.105, -0.077)         # robot0 transit 경로 위 정확한 위치

# Bimanual 시나리오용
TRANSIT_Z_OFFSET = 0.06                    # transit 시 낮은 고도 (병 충돌 유도)
```

### 3. 컨트롤러/제어 파라미터

| 상수 | 원본 | 최종 | 이유 |
|---|---|---|---|
| `JOINT_TOL` | `0.08` | `0.12` | 충돌 시 controller settling 여유 |
| `GRIP_CLOSE_HOLD_STEPS` | `12` | `20` | milk slip 방지 (충분히 잡힘) |
| `POSTGRASP_LIFT_Z_OFFSET` | `0.35` | `0.15` | transit과 가까운 고도로 부드러운 motion |

### 4. BDDL 동적 장애물 추가

기존 BDDL 생성 코드에 obstacle 분기 추가:

```python
# obstacle_region, wine_bottle_obstacle_1 - wine_bottle_obstacle, On 절을
# ENABLE_OBSTACLE=True일 때만 BDDL에 삽입
obstacle_region_str = "..." if ENABLE_OBSTACLE else ""
obstacle_objects_str = "  wine_bottle_obstacle_1 - wine_bottle_obstacle" if ENABLE_OBSTACLE else ""
obstacle_init_str = "    (On wine_bottle_obstacle_1 main_table_obstacle_region)" if ENABLE_OBSTACLE else ""
```

### 5. Transit waypoints 추가 (병/팔 충돌 유도용)

원본은 `lift_qpos → preplace_qpos` 직접 전이 (ee가 호를 그려 obstacle 우회).
중간 cartesian waypoints 추가해 ee가 낮은 고도로 직선 이동하게 강제:

```python
milk_transit_xyz   = (milk xy, milk_z + TRANSIT_Z_OFFSET)
mid_transit_xyz    = ((milk+trash)/2 xy, transit_z)
trash_transit_xyz  = (trash xy, transit_z)

# IK로 각 waypoint qpos 풀고 sequence에 삽입
mid_transit_qpos      = solve_ik_for_pose(...)
preplace_transit_qpos = solve_ik_for_pose(...)

# post_grasp_sequence에 transit_mid, transit_to_trash 추가
```

### 6. Robot1 bimanual trajectory (옵션 3)

`lock_robot1_pose()` 호출 모두 제거하고 robot1도 IK 기반 scripted policy 부여:

```python
# 시작은 retract 자세로 (robot0와 ee 충돌 방지)
ROBOT1_INITIAL_RETRACTED = np.array([0.0, -1.3, 0.0, -2.35, 0.0, 1.0, 0.785])

# milk_2 → trash_can_1 8-waypoint IK trajectory
r1_pregrasp_qpos = solve_ik_for_pose(env.env.sim, robot1, r1_pregrasp_xyz, ...)
r1_grasp_qpos    = ...
... (8개 waypoint)

# 시간 기반 keyframes (step → qpos + gripper)
R1_KEYFRAMES = [
    (0,   r1_home_qpos,     GRIPPER_OPEN),
    (35,  r1_pregrasp_qpos, GRIPPER_OPEN),
    (60,  r1_grasp_qpos,    GRIPPER_OPEN),
    (75,  r1_grasp_qpos,    GRIPPER_CLOSE),
    (95,  r1_postlift_qpos, GRIPPER_CLOSE),
    (110, r1_transit_qpos,  GRIPPER_CLOSE),
    (130, r1_mid_qpos,      GRIPPER_CLOSE),     # 중앙 통과 — 충돌 시점
    (150, r1_pretrash_qpos, GRIPPER_CLOSE),
    (170, r1_preplace_qpos, GRIPPER_CLOSE),
    (185, r1_place_qpos,    GRIPPER_CLOSE),
    (200, r1_place_qpos,    GRIPPER_OPEN),
]
def get_r1_target(step_t):
    """현재 step에서 보간된 target qpos + gripper."""
    ...
```

`make_dual_action`이 이제 robot1_action을 받음:

```python
def make_dual_action(robot0_action, robot1_action=None):
    if robot1_action is None:
        return list(robot0_action) + list(ROBOT1_PASSIVE_QPOS) + [ROBOT1_PASSIVE_GRIPPER_ACTION]
    return list(robot0_action) + list(robot1_action)
```

main loop에서 매 step robot1_action 계산 후 concat:

```python
r1_target_qpos, r1_gripper = get_r1_target(len(actions))
r1_cur = env.env.sim.data.qpos[robot1._ref_joint_pos_indexes].copy()
robot1_action = make_joint_position_action(r1_target_qpos, r1_cur, gripper_cmd=r1_gripper)
action = make_dual_action(robot0_action, robot1_action)
```

### 7. Stuck 자동 종료 로직

병/팔 충돌 시 robot이 같은 자세로 멈춰있는데 영상이 길게 늘어지는 문제 해결:

```python
last_jd = -1.0
stuck_count = 0
post_stuck_count = 0
STUCK_THRESHOLD = 25       # 연속 N step jd 변화 < 0.01이면 stuck
POST_STUCK_FRAMES = 30     # stuck 후 추가로 보여줄 frame (병 토플 물리)

# main loop 안:
in_transit_phase = phase in ("transit_mid", "transit_to_trash", "move_preplace_mid", "move_preplace")
if in_transit_phase and _jd > 0:
    if last_jd > 0 and abs(_jd - last_jd) < 0.01:
        stuck_count += 1
    else:
        stuck_count = 0
    last_jd = _jd
if stuck_count >= STUCK_THRESHOLD:
    post_stuck_count += 1
    if post_stuck_count >= POST_STUCK_FRAMES:
        done = True
```

### 8. 디버그 로깅 강화

매 step 출력에 ee xyz + bottle pos 추가:

```python
print(f"step={...} phase={...} jd={...} ee={ee_xyz} bottle={bottle_pos}")
```

env reset 직후 모든 객체 위치 dump:

```python
print("=== Scene objects ===")
for obj_name in obs:
    if obj_name.endswith("_pos"):
        print(f"  {obj_name}: {...}")
```

### 9. 경로 패치 (로컬 환경)

```python
# 원본: sys.path.insert(0, os.path.expanduser("~/RobotLearning_9/vlsa-aegis/safelibero"))
sys.path.insert(0, os.path.expanduser("/home/hannuri/works/work/RobotLearning_9-main/vlsa-aegis/safelibero"))
```

(민제 머신에 가져갈 땐 이 줄을 다시 `~/RobotLearning_9/...` 로 되돌려야 함)

---

## 신규 영상

| 파일 | 시나리오 | 결과 | 길이 |
|---|---|---|---|
| `AUTO_SCENE.mp4` | robot0 단독 milk → trash | success=True | 17초 |
| `AUTO_SCENE_OBSTACLE.mp4` | robot0 transit 중 wine bottle 충돌 (VLSA 미적용) | success=False, 병이 18cm 밀림 + 기울어짐 | 18초 |
| `AUTO_SCENE_ARM_COLLIDE.mp4` | 정적 robot1 팔이 robot0 경로 차단 | success=False, robot0 stuck | 19초 |
| `AUTO_SCENE_BIMANUAL.mp4` | 양쪽 robot 동시 milk → 같은 trash 수렴 | success=False, 중앙 충돌 | 40초 (다듬기 필요) |

---

## 알려진 이슈 / TODO

1. **Bimanual 영상 길이** — stuck 감지가 robot0 jd만 보는데 두 robot 비비대면서 jd 변동 → 400 step 다 돌음. `MAX_STEPS=220`으로 줄이거나 양쪽 모두 stuck 감지하면 짧아짐
2. **VLSA 적용 비교 미작성** — 위 시나리오들에 VLSA layer 켜서 회피 trajectory 비교 영상 필요 (다음 단계)
3. **데이터 수집 스크립트 미작성** — TODO.md의 LeRobot v3 포맷 dump 항목 (이 단계에서 만든 trajectory들을 episode로 dump)
4. **하드코딩 경로** — `auto_scene.py`의 sys.path가 `/home/hannuri/works/work/...`로 박혀있어 다른 머신에선 수정 필요

---

## 핵심 인사이트 (개발 중 발견)

1. **Joint-space 보간 trajectory는 직선이 아닌 호** — straight-line midpoint에 obstacle 둬도 robot이 자연스럽게 우회. 충돌 강제하려면 실제 ee 경로를 step-by-step 측정 후 그 위에 obstacle 배치 + transit z를 obstacle 본체 collision 범위 안으로 낮춰야 함

2. **Obstacle/Robot collision으로 stuck시 jd 진동** — robot이 obstacle을 살살 미는 사이 jd가 0.629/0.633 사이에서 진동. 임계값 0.002 → 0.01로 완화해야 stuck 감지

3. **두 opposing Panda의 home pose ee가 거의 동일 위치** — 기본 `HOME_QPOS` 그대로 두면 두 robot ee가 (0, 0, 1.3) 근처에 함께 있어 시작부터 충돌. robot1을 retract해서 시작해야 함

4. **wine_bottle_obstacle 본체 z 범위가 0.97~1.088** — 그보다 위로 transit하면 신경(neck, 1.05~1.22)에만 살짝 닿아 가벼운 push만 발생. 본체 정통 충돌 위해 transit z=1.03~1.05 필요
