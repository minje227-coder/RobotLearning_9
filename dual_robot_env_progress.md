# Dual Robot Tabletop Environment 진행사항

## 개요
vlsa-aegis(SafeLIBERO) 위에 두 Panda 로봇이 테이블 양 끝에서 마주보는 환경을 구성했다.  
싱글/듀얼 모드 모두 동일한 preview 스크립트로 조작 가능하다.

---

## 생성/수정 파일

### 신규
| 파일 | 설명 |
|------|------|
| `Data generation/preview_dual.py` | 환경 렌더링·MP4 저장 스크립트 (CLI 인수로 전부 조정) |
| `Data generation/dual_scene.bddl` | 기본 BDDL (실행 시 동적으로 덮어씀) |
| `vlsa-aegis/.../problems/dual_tabletop_manipulation.py` | 새 Problem 클래스 |
| `vlsa-aegis/.../robots/opposing_mounted_panda.py` | 반대편 로봇 모델 (미사용, 참고용) |

### 수정
| 파일 | 내용 |
|------|------|
| `problems/__init__.py` | dual_tabletop_manipulation import 추가 |
| `problems/libero_tabletop_manipulation.py` | `table_full_size` kwargs 버그 수정 |
| `regions/__init__.py` | `REGION_SAMPLERS`에 `libero_dual_tabletop_manipulation` 추가 |

---

## 핵심 구현

### LIBERO_Dual_Tabletop_Manipulation 클래스
- `_check_robot_configuration`: 로봇 1~2개 모두 허용
- `_load_model`: robot1을 테이블 반대편(`x = 0.16 + table_length/2`, rot=π)에 배치
- `_setup_camera` → `_add_side_tables_to_arena`: **mujoco_arena 단계**에서 side table 추가
  - visual geom: `group=1`, `material=table_texture` (기존 테이블과 동일 텍스처)
  - collision geom: `group=0`

### 모듈 레벨 설정 변수 (`dual_tabletop_manipulation.py`)
```python
SIDE_DEPTH    = None   # x 방향 깊이, None → table_length/2
SIDE_WIDTH    = 0.5    # y 방향 너비
SIDE_Y_OFFSET = None   # main_table 중심에서 y 거리, None → 자동
SIDE_X_OFFSET = 0.0    # 로봇 기준 x 추가 오프셋
```
env 생성 **전에** 덮어쓰면 반영된다.

---

## 삽질 기록

| 문제 | 원인 | 해결 |
|------|------|------|
| `KeyError: libero_dual_tabletop_manipulation` | `__init__.py` 순환 import | dual 모듈 내 중복 import 제거 |
| `KeyError` in REGION_SAMPLERS | REGION_SAMPLERS에 dual 미등록 | `regions/__init__.py`에 추가 |
| side table XML에는 있는데 안 보임 | `self.model` 완성 후 추가 → MuJoCo 컴파일 누락 | `_setup_camera` 훅에서 arena 단계에 추가 |
| side table sim에 있는데 렌더링 안됨 | geom group=0 (collision only) | visual geom을 group=1로 추가 |
| birdview에서 안 보임 | fovy=45°, z=3.0 → y=±1.24m 한계 | cam_pos z=5.0으로 조정 |

---

## preview_dual.py 사용법

```bash
# 로봇 2개 (기본값)
python preview_dual.py 2

# 로봇 1개
python preview_dual.py 1

# side table 크기/위치 조정
python preview_dual.py 2 --side-depth 0.4 --side-width 0.8 --side-y-offset 1.3 --side-x-offset 0.1

# 바구니/우유 위치 조정 (table 중심 기준, x_min x_max)
python preview_dual.py 2 --basket-y -0.6 -0.4

# 우유 로봇 기준으로 배치
python preview_dual.py 2 --milk-ref robot --milk-x 0.1 0.2 --milk-y -0.1 0.1
```

### 전체 인수 목록
| 인수 | 기본값 | 설명 |
|------|--------|------|
| `robots` | 필수 (1 or 2) | 로봇 수 |
| `--side-depth` | 0.8 | side table x 깊이 (m) |
| `--side-width` | 0.7 | side table y 너비 (m) |
| `--side-y-offset` | 0.5 | main_table 중심에서 y 거리 (m) |
| `--side-x-offset` | 0.0 | 로봇 기준 x 추가 오프셋 (m) |
| `--basket-x` | -0.0 -0.0 | 바구니 x 범위 (table 기준) |
| `--basket-y` | -0.3 -0.3 | 바구니 y 범위 (table 기준) |
| `--milk-x` | -0.0 -0.0 | 우유 x 범위 |
| `--milk-y` | 0.4 0.4 | 우유 y 범위 |
| `--milk-ref` | robot | 우유 기준점 (table / robot) |

---

## 좌표계 참고
- main_table 중심 = (0, 0)
- robot0: x = -(0.16 + table_length/2) ≈ -0.46
- robot1: x = +(0.16 + table_length/2) ≈ +0.46
- table_full_size = (0.6, 1.7, 0.05) → y: -0.85 ~ +0.85
- birdview 카메라: pos=(-0.2, 0.0, 5.0)
