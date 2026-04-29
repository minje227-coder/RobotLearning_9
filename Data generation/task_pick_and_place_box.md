# Pick & Place into Box — Task 구성 자산 정리

로봇팔로 객체를 집어서 중간의 박스/컨테이너에 넣는 task 구성용.

## 📦 박스 / 컨테이너 (목표 지점)

| 이름 | 카테고리 | 비고 |
|---|---|---|
| `white_storage_box` | Turbosquid | 뚜껑 없는 흰 수납함 ← 가장 적합 |
| `box_base` | Obstacle | 단순한 박스 베이스 |
| `box_small_base` | Obstacle | 작은 박스 |
| `basket` | Stable Scanned | 바구니 (SafeLIBERO에서도 쓰임) |

## 🎯 집을 객체 (Source Object)

### HOPE 식료품 — 직육면체라 그리퍼로 잡기 쉬움
`alphabet_soup`, `milk`, `orange_juice`, `chocolate_pudding`, `bbq_sauce`, `cream_cheese`, `tomato_sauce`

### Turbosquid 생활용품 — 다양한 형태
`porcelain_mug`, `red_coffee_mug`, `white_yellow_mug`, `wine_bottle`, `moka_pot`

### Stable Scanned 식기류 — 둥근 형태
`akita_black_bowl`, `white_bowl`, `red_bowl`

## 🤖 로봇 + 환경

- **로봇:** `mounted_panda` (테이블 고정형이 이 task에 적합)
- **환경:** `table_arena` (단순하고 깔끔) 또는 `kitchen_arena`

## ✅ 추천 조합

| 항목 | 선택 |
|---|---|
| 집을 것 | `milk` |
| 목표 컨테이너 | `white_storage_box` |
| 환경 | `table_arena` |
| 로봇 | `mounted_panda` |

**Task description 예시:**
> *"Pick up the milk and place it in the white storage box."*
