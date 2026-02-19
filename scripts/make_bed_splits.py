#!/usr/bin/env python3
"""
BED-BPP 원본 JSON을 order-id 기준으로 train/val/test로 고정 분할한다.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic BED-BPP splits.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("GOPT/bed-bpp_v1.json"),
        help="원본 BED-BPP JSON 경로",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("GOPT/data/bed_bpp/splits"),
        help="분할 JSON 출력 디렉토리",
    )
    parser.add_argument("--seed", type=int, default=42, help="분할 셔플 seed")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="train 비율")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="val 비율")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="test 비율")
    return parser.parse_args()


def validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"split ratio 합이 1.0이어야 합니다. (현재: {total})")
    if min(train, val, test) < 0:
        raise ValueError("split ratio는 음수가 될 수 없습니다.")


def load_orders(source: Path) -> Dict[str, dict]:
    if not source.exists():
        raise FileNotFoundError(f"원본 데이터셋 파일을 찾을 수 없습니다: {source}")
    with source.open("r", encoding="utf-8") as f:
        orders = json.load(f)
    if not isinstance(orders, dict):
        raise TypeError("BED-BPP JSON 최상위 구조는 dict여야 합니다.")
    return orders


def split_order_ids(
    order_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[list[str], list[str], list[str]]:
    # 재현 가능한 분할을 위해 고정 seed RNG만 사용한다.
    rng = random.Random(seed)
    ids = list(order_ids)
    rng.shuffle(ids)

    total = len(ids)
    val_count = int(total * val_ratio)
    test_count = int(total * (1.0 - train_ratio - val_ratio))
    train_count = total - val_count - test_count

    train_ids = ids[:train_count]
    val_ids = ids[train_count:train_count + val_count]
    test_ids = ids[train_count + val_count:]
    return train_ids, val_ids, test_ids


def build_subset(orders: Dict[str, dict], ids: list[str]) -> Dict[str, dict]:
    # 결과 JSON의 키 순서를 고정해 후속 diff/검증을 쉽게 만든다.
    return {order_id: orders[order_id] for order_id in sorted(ids)}


def target_counts(subset: Dict[str, dict]) -> Dict[str, int]:
    counts = Counter()
    for order in subset.values():
        target = order.get("properties", {}).get("target", "unknown")
        counts[target] += 1
    return dict(counts)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    orders = load_orders(args.source)

    order_ids = sorted(orders.keys())
    train_ids, val_ids, test_ids = split_order_ids(
        order_ids=order_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_subset = build_subset(orders, train_ids)
    val_subset = build_subset(orders, val_ids)
    test_subset = build_subset(orders, test_ids)

    train_path = args.out_dir / "bed_bpp_v1_train.json"
    val_path = args.out_dir / "bed_bpp_v1_val.json"
    test_path = args.out_dir / "bed_bpp_v1_test.json"
    manifest_path = args.out_dir / "split_manifest.json"

    write_json(train_path, train_subset)
    write_json(val_path, val_subset)
    write_json(test_path, test_subset)

    manifest = {
        "source": str(args.source),
        "seed": int(args.seed),
        "ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "counts": {
            "train": len(train_subset),
            "val": len(val_subset),
            "test": len(test_subset),
        },
        "target_counts": {
            "train": target_counts(train_subset),
            "val": target_counts(val_subset),
            "test": target_counts(test_subset),
        },
    }
    write_json(manifest_path, manifest)

    print(f"train: {len(train_subset)} -> {train_path}")
    print(f"val:   {len(val_subset)} -> {val_path}")
    print(f"test:  {len(test_subset)} -> {test_path}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
