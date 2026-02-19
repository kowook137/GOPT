#!/usr/bin/env python3
"""
BED 환경 스모크 테스트:
- 관측 key/shape/dtype/NaN 점검
- step/reset 루프 점검
- 주문 소진(StopIteration 처리 -> order_done) 점검
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import gymnasium as gym


SCRIPT_DIR = Path(__file__).resolve().parent
GOPT_DIR = SCRIPT_DIR.parent
if str(GOPT_DIR) not in sys.path:
    sys.path.append(str(GOPT_DIR))

from tools import registration_envs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for BED env.")
    parser.add_argument(
        "--env-id",
        type=str,
        default="OnlinePackBed-v1",
        help="테스트할 gym env id",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("GOPT/data/bed_bpp/splits/bed_bpp_v1_val.json"),
        help="검증에 사용할 BED split JSON",
    )
    parser.add_argument("--episodes", type=int, default=5, help="랜덤 실행 에피소드 수")
    parser.add_argument("--max-steps", type=int, default=128, help="에피소드당 최대 스텝")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 seed")
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--max-boxes", type=int, default=128)
    return parser.parse_args()


def _assert_obs(obs: Dict[str, np.ndarray], max_boxes: int, max_candidates: int) -> None:
    required = {
        "boxes_array",
        "next_box",
        "bin_dims",
        "candidate_boxes",
        "candidate_positions",
        "candidate_oris",
        "mask",
    }
    missing = required - set(obs.keys())
    if missing:
        raise AssertionError(f"관측 key 누락: {sorted(missing)}")

    if obs["boxes_array"].shape != (max_boxes, 10):
        raise AssertionError(f"boxes_array shape 불일치: {obs['boxes_array'].shape}")
    if obs["next_box"].shape != (2, 3):
        raise AssertionError(f"next_box shape 불일치: {obs['next_box'].shape}")
    if obs["bin_dims"].shape != (3,):
        raise AssertionError(f"bin_dims shape 불일치: {obs['bin_dims'].shape}")
    if obs["candidate_boxes"].shape != (max_candidates, 3):
        raise AssertionError(f"candidate_boxes shape 불일치: {obs['candidate_boxes'].shape}")
    if obs["candidate_positions"].shape != (max_candidates, 3):
        raise AssertionError(f"candidate_positions shape 불일치: {obs['candidate_positions'].shape}")
    if obs["candidate_oris"].shape != (max_candidates,):
        raise AssertionError(f"candidate_oris shape 불일치: {obs['candidate_oris'].shape}")
    if obs["mask"].shape != (max_candidates,):
        raise AssertionError(f"mask shape 불일치: {obs['mask'].shape}")

    if obs["boxes_array"].dtype != np.float32:
        raise AssertionError(f"boxes_array dtype 불일치: {obs['boxes_array'].dtype}")
    if obs["mask"].dtype != np.bool_:
        raise AssertionError(f"mask dtype 불일치: {obs['mask'].dtype}")

    float_keys = ["boxes_array", "next_box", "bin_dims", "candidate_boxes", "candidate_positions"]
    for key in float_keys:
        if np.isnan(obs[key]).any():
            raise AssertionError(f"{key}에 NaN 포함")


def _make_env(
    env_id: str,
    dataset_path: Path,
    max_candidates: int,
    max_boxes: int,
    seed: int,
) -> gym.Env:
    return gym.make(
        env_id,
        data_type="bed",
        bed_dataset_path=str(dataset_path),
        bed_dataset_seed=seed,
        bed_dataset_shuffle_orders=True,
        enable_rotation=True,
        action_scheme="EMS",
        max_candidates=max_candidates,
        max_boxes=max_boxes,
        max_points=0,
        item_set=[],
    )


def run_random_smoke(args: argparse.Namespace) -> Tuple[int, int, int]:
    env = _make_env(
        env_id=args.env_id,
        dataset_path=args.dataset_path,
        max_candidates=args.max_candidates,
        max_boxes=args.max_boxes,
        seed=args.seed,
    )

    done_count = 0
    invalid_action_count = 0
    order_done_count = 0

    for _ in range(args.episodes):
        obs, _ = env.reset()
        _assert_obs(obs, args.max_boxes, args.max_candidates)

        for _ in range(args.max_steps):
            valid = np.where(obs["mask"])[0]
            action = int(random.choice(valid.tolist())) if len(valid) > 0 else 0
            obs, _, terminated, truncated, info = env.step(action)
            _assert_obs(obs, args.max_boxes, args.max_candidates)

            if bool(info.get("invalid_action", False)):
                invalid_action_count += 1
            if bool(info.get("order_done", False)):
                order_done_count += 1

            if terminated or truncated:
                done_count += 1
                break

    env.close()
    return done_count, invalid_action_count, order_done_count


def run_stop_iteration_smoke(args: argparse.Namespace) -> bool:
    # 주문 1개/아이템 1개 데이터로 강제 소진 케이스를 만든다.
    tiny = {
        "00000001": {
            "item_sequence": {
                "1": {
                    "article": "tiny-article",
                    "id": "1",
                    "product_group": "tiny",
                    "length/mm": 100,
                    "width/mm": 100,
                    "height/mm": 100,
                    "weight/kg": 1.0,
                    "sequence": 1,
                }
            },
            "properties": {
                "id": "00000001",
                "order_nr": 1,
                "type": "bed",
                "target": "euro-pallet",
            },
        }
    }

    with tempfile.TemporaryDirectory(prefix="bed-smoke-") as tmp:
        tmp_path = Path(tmp) / "tiny.json"
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(tiny, f, ensure_ascii=False, indent=2)

        env = _make_env(
            env_id=args.env_id,
            dataset_path=tmp_path,
            max_candidates=args.max_candidates,
            max_boxes=args.max_boxes,
            seed=args.seed,
        )
        obs, _ = env.reset()
        _assert_obs(obs, args.max_boxes, args.max_candidates)

        valid = np.where(obs["mask"])[0]
        if len(valid) == 0:
            env.close()
            raise AssertionError("tiny 데이터에서 유효 후보가 없어 StopIteration 점검을 진행할 수 없습니다.")

        obs2, _, terminated, truncated, info = env.step(int(valid[0]))
        _assert_obs(obs2, args.max_boxes, args.max_candidates)
        env.close()
        return bool(terminated) and (not bool(truncated)) and bool(info.get("order_done", False))


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    registration_envs()

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"dataset 파일이 없습니다: {args.dataset_path}")

    done_count, invalid_action_count, order_done_count = run_random_smoke(args)
    stop_ok = run_stop_iteration_smoke(args)

    print(f"episodes={args.episodes}")
    print(f"done_count={done_count}")
    print(f"invalid_action_count={invalid_action_count}")
    print(f"order_done_count={order_done_count}")
    print(f"stop_iteration_handling_ok={int(stop_ok)}")

    passed = done_count > 0 and stop_ok
    print(f"overall_passed={int(passed)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
