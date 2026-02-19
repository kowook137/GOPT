# BED-BPP 실험 프로토콜 고정안 (GOPT)

## 1. 목적
- 논문/보고 시 "BED-BPP 데이터셋 사용" 주장을 재현 가능하게 만든다.
- 데이터, seed, 지표, 비교 조건을 고정해 실험 임의성을 줄인다.

## 2. 데이터 고정
- 원본: `GOPT/bed-bpp_v1.json`
- split:
  - train: `GOPT/data/bed_bpp/splits/bed_bpp_v1_train.json`
  - val: `GOPT/data/bed_bpp/splits/bed_bpp_v1_val.json`
  - test: `GOPT/data/bed_bpp/splits/bed_bpp_v1_test.json`
- split seed: `42` 고정
- 해상도: `1mm` 고정 (`env.resolution_mm: 1`)

## 3. 학습/평가 seed 고정
- 권장 seed 리스트: `[5, 11, 17, 23, 29]`
- 각 seed별로 동일 split, 동일 하이퍼파라미터로 1회 학습 후 test 평가
- 최종 보고는 평균과 표준편차를 함께 제시

## 4. 고정 하이퍼파라미터
- 후보 수: `env.max_candidates=80`
- 박스 버퍼: `env.max_boxes=300`
- 포인트클라우드: `env.max_points=0`
- 회전 허용: `env.rot=True`
- action scheme: `env.scheme=EMS`
- 알고리즘: `PPO`
- 그 외 학습 하이퍼파라미터는 `GOPT/cfg/config.yaml` 값을 고정

## 5. 보고 지표 고정
- 주문 완료율: `order_completion_rate`
- 평균 공간 활용률: `average space utilization`
- 평균 적재 아이템 수: `average put item number`
- 활용률 표준편차: `ratio_std`

보조 분석(권장):
- target별(`euro-pallet`, `rollcontainer`) 지표 분리
- 주문별 utilization 분포(사분위/히스토그램)

## 6. Baseline(RS) 비교 조건 통제
- 동일 train/val/test 주문 집합 사용
- 동일 seed 리스트 사용
- 동일 모델 크기/학습 step/optimizer 설정 사용
- 동일 candidate budget(`max_candidates`) 사용
- 비교군 간 변경점은 "데이터 소스 및 관측 구성"으로 제한

## 7. 실행 순서
1. split 생성/검증: `make_bed_splits.py`, `validate_bed_fidelity.py`
2. smoke 테스트: `smoke_bed_env.py`
3. seed별 학습 실행
4. seed별 test 평가 실행
5. 지표 집계(mean/std) 및 표/그림 생성

## 8. 재현 명령 템플릿
학습:
```bash
python GOPT/ts_train.py --config cfg/config.yaml --device 0
```

평가:
```bash
python GOPT/ts_test.py --config cfg/config.yaml --device 0 --ckp /path/to/policy_step_final.pth
```

주의:
- 현재 구성은 `cfg/config.yaml`의 `seed`를 읽는다.
- seed sweep 시에는 각 실행 전에 `seed`를 명시적으로 변경해 기록을 남긴다.
