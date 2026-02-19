import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .binCreator import BoxCreator


class BedBoxCreator(BoxCreator):
    """
    BED-BPP JSON(order 단위)을 순차적으로 재생하는 박스 생성기.
    """

    def __init__(
        self,
        data_name: str,
        dataset_seed: int = 42,
        dataset_shuffle_orders: bool = True,
    ):
        super().__init__()
        self.data_name = str(data_name)
        self.dataset_seed = int(dataset_seed)
        self.dataset_shuffle_orders = bool(dataset_shuffle_orders)

        self._rng = random.Random(self.dataset_seed)
        self._orders: Dict[str, dict] = {}
        self._order_ids: List[str] = []
        self._order_cursor = 0

        self.current_order_id: Optional[str] = None
        self.current_target: Optional[str] = None
        self._current_items: List[Tuple[int, int, int, str]] = []
        self._current_item_cursor = 0

        self.recorder: List[Tuple[int, int, int, str]] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        path = Path(self.data_name)
        if not path.exists():
            raise FileNotFoundError(f"BED dataset file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            raw_orders = json.load(f)

        if not isinstance(raw_orders, dict):
            raise TypeError("BED dataset top-level JSON must be dict(order_id -> order).")

        parsed_orders: Dict[str, dict] = {}
        for order_id, order in raw_orders.items():
            if not isinstance(order, dict):
                continue
            item_sequence = order.get("item_sequence", {})
            if not isinstance(item_sequence, dict):
                item_sequence = {}

            # item_sequence 키를 정수 기준으로 정렬해 원본 주문 순서를 보존한다.
            ordered_keys = sorted(item_sequence.keys(), key=lambda x: int(x))
            items: List[Tuple[int, int, int, str]] = []
            for key in ordered_keys:
                item = item_sequence.get(key, {})
                w = int(item.get("length/mm", 0))
                d = int(item.get("width/mm", 0))
                h = int(item.get("height/mm", 0))
                box_type = str(item.get("product_group", "unknown"))
                items.append((w, d, h, box_type))

            parsed_orders[str(order_id)] = {
                "target": order.get("properties", {}).get("target", "euro-pallet"),
                "items": items,
            }

        self._orders = parsed_orders
        self._order_ids = sorted(self._orders.keys())
        if self.dataset_shuffle_orders:
            self._rng.shuffle(self._order_ids)

    def start_new_episode(self) -> dict:
        if not self._order_ids:
            raise RuntimeError("No orders loaded in BED dataset.")

        if self._order_cursor >= len(self._order_ids):
            self._order_cursor = 0
            if self.dataset_shuffle_orders:
                # 한 바퀴를 모두 돈 뒤에는 같은 seed RNG 상태로 다시 섞는다.
                self._rng.shuffle(self._order_ids)

        order_id = self._order_ids[self._order_cursor]
        self._order_cursor += 1
        order = self._orders[order_id]

        self.current_order_id = order_id
        self.current_target = str(order.get("target", "euro-pallet"))
        self._current_items = list(order.get("items", []))
        self._current_item_cursor = 0
        self.box_list.clear()
        self.recorder = []

        return {
            "order_id": self.current_order_id,
            "target": self.current_target,
            "num_items": len(self._current_items),
        }

    def has_remaining_items(self) -> bool:
        return self._current_item_cursor < len(self._current_items)

    def reset(self, index=None):
        if index is not None:
            idx = int(index) % max(1, len(self._order_ids))
            self._order_cursor = idx
        return self.start_new_episode()

    def generate_box_size(self, **kwargs):
        if self.current_order_id is None:
            self.start_new_episode()

        if not self.has_remaining_items():
            raise StopIteration("Current BED order has no remaining items.")

        w, d, h, box_type = self._current_items[self._current_item_cursor]
        self._current_item_cursor += 1
        self.box_list.append((w, d, h))
        self.recorder.append((w, d, h, box_type))

    def preview(self, length):
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])
