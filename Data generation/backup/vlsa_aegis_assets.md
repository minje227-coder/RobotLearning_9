# vlsa-aegis 사용 가능 자산 정리

vlsa-aegis는 LIBERO 기반(safelibero)으로 만든 안전성 평가 벤치마크.
경로: `/home/minje/RobotLearning_9/vlsa-aegis/safelibero/libero/libero/`

## 🤖 로봇 팔
경로: `envs/robots/`

- **mounted_panda** — 테이블 위에 마운트된 Franka Panda
- **on_the_ground_panda** — 바닥에 놓인 Franka Panda

> LIBERO 기본은 Panda 한 종류만 지원

## 🏞️ 환경 (Arena / 배경)
경로: `envs/arenas/`

| Arena | 설명 |
|---|---|
| `table_arena` | 일반 테이블탑 |
| `kitchen_arena` | 부엌 (스토브/캐비닛 포함) |
| `living_room_arena` | 거실 (소파/테이블) |
| `study_arena` | 서재 (책장/벽 장식) |
| `coffee_table_arena` | 커피 테이블 |
| `empty_arena` | 빈 공간 |

**스타일 변형 (텍스처):** `base`, `blue`, `warm`, `coffee` — 테이블·바닥에 적용

**배경 소품 (assets/scenes):**
`kitchen_background_{hot_pot, pot, stove}`, `office_book_shelf`, `wooden_shelf`, `floor_lamp`, `plant`, `wall_decoration`, `study_wall_painting`, `living_room_table`, `short_coffee_table`, `fridge`, `desk`

## 📦 객체

### Stable Scanned (식기류)
경로: `assets/stable_scanned_objects/`

`akita_black_bowl`, `white_bowl`, `red_bowl`, `plate`, `basket`, `chefmate_8_frypan`, `glazed_rim_porcelain_ramekin`, `simple_rack`, `basin_faucet_base`, `basin_faucet_movable`

### Stable HOPE (식료품 패키지)
경로: `assets/stable_hope_objects/`

`alphabet_soup`, `bbq_sauce`, `butter`, `chocolate_pudding`, `cookies`, `cream_cheese`, `ketchup`, `macaroni_and_cheese`, `milk`, `orange_juice`, `popcorn`, `salad_dressing`, `new_salad_dressing`, `tomato_sauce`

### Turbosquid (생활용품)
경로: `assets/turbosquid_objects/`

`black_book`, `yellow_book`, `bowl_drainer`, `desk_caddy`, `moka_pot`, `porcelain_mug`, `red_coffee_mug`, `white_yellow_mug`, `wine_bottle`, `wine_rack`, `wine_rack_stand`, `white_storage_box`, `wooden_tray`, `wooden_shelf`, `wooden_two_layer_shelf`, `wooden_cabinet_base`, `dining_set_group`

### Articulated (관절형 / 개폐형)
경로: `assets/articulated_objects/`

`flat_stove`, `microwave`, `short_cabinet`, `wooden_cabinet`, `white_cabinet`, `slide_cabinet`, `short_fridge`, `basin_faucet`, `window`

### Obstacle (장애물 변형 — SafeLIBERO 핵심)
경로: `assets/obstacle_objects/`

위 객체들의 `_obstacle` 버전:
`alphabet_soup_obstacle`, `basin_faucet_base_obstacle`, `basket_obstacle`, `bbq_sauce_obstacle`, `box_base`, `box_small_base`, `butter_obstacle`, `chefmate_8_frypan_obstacle`, `chocolate_pudding_obstacle`, `cream_cheese_obstacle`, `ketchup_obstacle`, `milk_obstacle`, `milk_small_obstacle`, `moka_pot_obstacle`, `moka_pot_small_obstacle`, `new_salad_dressing_obstacle`, `orange_juice_obstacle`, `popcorn_obstacle`, `red_coffee_mug_obstacle`, `simple_rack_obstacle`, `white_storage_box_obstacle`, `wine_bottle_obstacle`, `wine_bottle_small_obstacle`, `yellow_book_obstacle`

## 📁 객체 정의 파이썬 모듈
경로: `envs/objects/`

- `google_scanned_objects.py` — stable scanned 객체 클래스
- `hope_objects.py` — HOPE 식료품 클래스
- `turbosquid_objects.py` — turbosquid 객체 클래스
- `articulated_objects.py` — 관절형 객체 클래스
- `obstacle_objects.py` — 장애물 객체 클래스
- `target_zones.py` — 목표 영역
- `site_object.py` — site 마커

## 📝 BDDL 태스크 정의
경로: `libero/bddl_files/safelibero_{spatial, goal, object, long}/`

각 suite별 task의 `.bddl` (객체/조건) 과 `.pruned_init` (50 episode 초기 상태) 파일 존재.
