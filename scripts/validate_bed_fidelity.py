#!/usr/bin/env python3
"""
BED-BPP 원본 대비 split/로더 충실도를 검증하고 JSON/CSV 리포트를 저장한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
GOPT_DIR = SCRIPT_DIR.parent
# 저장소 루트에서 실행해도 GOPT 패키지(import envs)가 잡히도록 경로를 보강한다.
if str(GOPT_DIR) not in sys.path:
    sys.path.append(str(GOPT_DIR))

# BedBoxCreator를 통해 실제 로더 출력 순서/값도 함께 검증한다.
from envs.Packing.bedCreator import BedBoxCreator


@dataclass
class SplitStats:
    split: str
    orders: int = 0
    items: int = 0
    missing_in_source: int = 0
    source_order_missing_in_split: int = 0
    item_count_mismatch: int = 0
    item_key_order_mismatch: int = 0
    item_dim_mismatch: int = 0
    item_type_mismatch: int = 0
    target_mismatch: int = 0
    creator_order_mismatch: int = 0
    creator_item_dim_mismatch: int = 0
    creator_item_type_mismatch: int = 0
    creator_stop_mismatch: int = 0
    passed: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BED split and loader fidelity.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("GOPT/bed-bpp_v1.json"),
        help="원본 BED JSON 경로",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("GOPT/data/bed_bpp/splits"),
        help="split JSON 디렉토리",
    )
    parser.add_argument(
        "--check-creator",
        action="store_true",
        help="BedBoxCreator 재생 결과까지 검증",
    )
    parser.add_argument(
        "--creator-seed",
        type=int,
        default=42,
        help="BedBoxCreator 검증 시 사용할 seed",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("GOPT/data/bed_bpp/validation/fidelity_report.json"),
        help="검증 리포트(JSON) 출력 경로",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("GOPT/data/bed_bpp/validation/fidelity_report.csv"),
        help="검증 리포트(CSV) 출력 경로",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"최상위 구조는 dict여야 합니다: {path}")
    return data


def extract_items(order: dict) -> Tuple[List[str], List[Tuple[int, int, int, str]]]:
    seq = order.get("item_sequence", {})
    if not isinstance(seq, dict):
        return [], []
    keys = sorted(seq.keys(), key=lambda x: int(x))
    items: List[Tuple[int, int, int, str]] = []
    for key in keys:
        item = seq.get(key, {})
        w = int(item.get("length/mm", 0))
        d = int(item.get("width/mm", 0))
        h = int(item.get("height/mm", 0))
        t = str(item.get("product_group", "unknown"))
        items.append((w, d, h, t))
    return keys, items


def compare_split_to_source(
    split_name: str,
    split_orders: Dict[str, dict],
    source_orders: Dict[str, dict],
) -> SplitStats:
    stats = SplitStats(split=split_name, orders=len(split_orders))

    for order_id, split_order in split_orders.items():
        if order_id not in source_orders:
            stats.missing_in_source += 1
            continue

        src_order = source_orders[order_id]
        src_keys, src_items = extract_items(src_order)
        split_keys, split_items = extract_items(split_order)
        stats.items += len(split_items)

        src_target = str(src_order.get("properties", {}).get("target", ""))
        split_target = str(split_order.get("properties", {}).get("target", ""))
        if src_target != split_target:
            stats.target_mismatch += 1

        if len(src_items) != len(split_items):
            stats.item_count_mismatch += 1

        if src_keys != split_keys:
            stats.item_key_order_mismatch += 1

        for src_item, split_item in zip(src_items, split_items):
            if src_item[:3] != split_item[:3]:
                stats.item_dim_mismatch += 1
            if src_item[3] != split_item[3]:
                stats.item_type_mismatch += 1

    source_only = set(source_orders.keys()) - set(split_orders.keys())
    stats.source_order_missing_in_split = len(source_only)

    mismatch_total = (
        stats.missing_in_source
        + stats.item_count_mismatch
        + stats.item_key_order_mismatch
        + stats.item_dim_mismatch
        + stats.item_type_mismatch
        + stats.target_mismatch
    )
    stats.passed = mismatch_total == 0
    return stats


def validate_creator_replay(
    stats: SplitStats,
    split_path: Path,
    split_orders: Dict[str, dict],
    creator_seed: int,
) -> None:
    # split 파일 자체를 로더에 넣고, 로더가 같은 주문/아이템을 같은 순서로 내는지 확인한다.
    creator = BedBoxCreator(
        data_name=str(split_path),
        dataset_seed=creator_seed,
        dataset_shuffle_orders=False,
    )
    expected_ids = sorted(split_orders.keys())

    for expected_order_id in expected_ids:
        meta = creator.start_new_episode()
        got_order_id = str(meta.get("order_id", ""))
        if got_order_id != expected_order_id:
            stats.creator_order_mismatch += 1

        _, expected_items = extract_items(split_orders[expected_order_id])
        for exp_w, exp_d, exp_h, exp_t in expected_items:
            creator.generate_box_size()
            got_w, got_d, got_h, got_t = creator.recorder[-1]

            if (int(got_w), int(got_d), int(got_h)) != (exp_w, exp_d, exp_h):
                stats.creator_item_dim_mismatch += 1
            if str(got_t) != exp_t:
                stats.creator_item_type_mismatch += 1

            creator.drop_box()

        # 주문이 끝난 뒤에는 StopIteration이 발생해야 주문 경계가 보장된다.
        try:
            creator.generate_box_size()
            stats.creator_stop_mismatch += 1
            creator.drop_box()
        except StopIteration:
            pass

    if (
        stats.creator_order_mismatch
        + stats.creator_item_dim_mismatch
        + stats.creator_item_type_mismatch
        + stats.creator_stop_mismatch
    ) > 0:
        stats.passed = False


def write_report_csv(path: Path, split_stats: List[SplitStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(split_stats[0]).keys()) if split_stats else list(asdict(SplitStats("empty")).keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in split_stats:
            writer.writerow(asdict(row))


def main() -> int:
    args = parse_args()

    source_orders = load_json(args.source)
    split_files = {
        "train": args.splits_dir / "bed_bpp_v1_train.json",
        "val": args.splits_dir / "bed_bpp_v1_val.json",
        "test": args.splits_dir / "bed_bpp_v1_test.json",
    }
    split_orders_map = {name: load_json(path) for name, path in split_files.items()}

    split_stats: List[SplitStats] = []
    all_split_ids = set()
    split_id_sets = {}
    for name, orders in split_orders_map.items():
        split_id_sets[name] = set(orders.keys())
        all_split_ids.update(orders.keys())

    for name, orders in split_orders_map.items():
        stats = compare_split_to_source(name, orders, source_orders)
        if args.check_creator:
            validate_creator_replay(stats, split_files[name], orders, args.creator_seed)
        split_stats.append(stats)

    overlap_count = (
        len(split_id_sets["train"] & split_id_sets["val"])
        + len(split_id_sets["train"] & split_id_sets["test"])
        + len(split_id_sets["val"] & split_id_sets["test"])
    )
    source_ids = set(source_orders.keys())
    missing_ids = sorted(source_ids - all_split_ids)
    extra_ids = sorted(all_split_ids - source_ids)

    report = {
        "source_path": str(args.source),
        "splits_dir": str(args.splits_dir),
        "check_creator": bool(args.check_creator),
        "creator_seed": int(args.creator_seed),
        "source_orders": len(source_orders),
        "source_items": sum(len(extract_items(o)[1]) for o in source_orders.values()),
        "partition_check": {
            "union_equals_source": len(missing_ids) == 0 and len(extra_ids) == 0,
            "split_overlap_count": overlap_count,
            "missing_ids_count": len(missing_ids),
            "extra_ids_count": len(extra_ids),
            "missing_ids_sample": missing_ids[:20],
            "extra_ids_sample": extra_ids[:20],
        },
        "splits": [asdict(s) for s in split_stats],
        "overall_passed": (
            overlap_count == 0
            and len(missing_ids) == 0
            and len(extra_ids) == 0
            and all(s.passed for s in split_stats)
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_report_csv(args.out_csv, split_stats)

    print(f"report json: {args.out_json}")
    print(f"report csv:  {args.out_csv}")
    print(f"overall_passed: {report['overall_passed']}")

    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
