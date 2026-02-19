# BED-BPP 적용 프로토콜 (GOPT, 1mm 무변환)

## 1. 목적
- 목표: `GOPT/bed-bpp_v1.json`를 GOPT 학습/평가에 엄밀하게 반영한다.
- 원칙: 원본 데이터의 단위/순서/타입 정보를 보존하고, 임의 변환으로 의미를 훼손하지 않는다.

## 2. 데이터 충실도 규칙
- 단위: `mm`를 그대로 사용한다.
- 스케일 변환: 금지한다.
- 반올림/내림/올림: 금지한다.
- 정규화: 모델 입력 직전(`model.py`)에서만 수행한다.
- 주문/아이템 순서: 원본 `item_sequence`를 정수 key 오름차순으로 유지한다.

원본 필드 매핑:

| BED 원본 필드 | 내부 표현 |
|---|---|
| `length/mm` | `w_mm` |
| `width/mm` | `d_mm` |
| `height/mm` | `h_mm` |
| `product_group` | `box_type` |
| `properties.target` | bin type (`euro-pallet`/`rollcontainer`) |

## 3. Split 고정 규칙
- 스크립트: `GOPT/scripts/make_bed_splits.py`
- 고정 seed: `42`
- 비율: `train=0.8`, `val=0.1`, `test=0.1`
- 현재 생성 결과: `train=8003`, `val=1000`, `test=1000`

산출물:
- `GOPT/data/bed_bpp/splits/bed_bpp_v1_train.json`
- `GOPT/data/bed_bpp/splits/bed_bpp_v1_val.json`
- `GOPT/data/bed_bpp/splits/bed_bpp_v1_test.json`
- `GOPT/data/bed_bpp/splits/split_manifest.json`

## 4. 환경 반영 규칙
- 모드: BED 전용 (`data_type='bed'`)
- target-to-bin 매핑:
  - `euro-pallet` -> `(1200, 800, bed_target_height)`
  - `rollcontainer` -> `(800, 700, bed_target_height)`
- 기본 높이: `bed_target_height=2000` (mm)
- 관측:
  - `boxes_array`, `next_box`, `bin_dims`, `candidate_*`, `mask`
  - 모두 mm 기반 raw 값으로 구성

## 5. 검증 절차

1) split 생성
```bash
python GOPT/scripts/make_bed_splits.py \
  --source GOPT/bed-bpp_v1.json \
  --out-dir GOPT/data/bed_bpp/splits \
  --seed 42
```

2) 원본 대비 충실도 검증
```bash
python GOPT/scripts/validate_bed_fidelity.py \
  --source GOPT/bed-bpp_v1.json \
  --splits-dir GOPT/data/bed_bpp/splits \
  --check-creator \
  --creator-seed 42
```

3) 환경 스모크 테스트
```bash
python GOPT/scripts/smoke_bed_env.py \
  --env-id OnlinePackBed-v1 \
  --dataset-path GOPT/data/bed_bpp/splits/bed_bpp_v1_val.json \
  --episodes 10 \
  --seed 42
```

## 6. 증빙 파일
- 충실도 리포트(JSON): `GOPT/data/bed_bpp/validation/fidelity_report.json`
- 충실도 리포트(CSV): `GOPT/data/bed_bpp/validation/fidelity_report.csv`
- split 메타데이터: `GOPT/data/bed_bpp/splits/split_manifest.json`
