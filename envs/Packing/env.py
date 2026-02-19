from typing import Dict, List, Optional

from .container import Container
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .bedCreator import BedBoxCreator

from render import VTKRender


class PackingEnv(gym.Env):
    def __init__(
        self,
        container_size=(10, 10, 10),
        item_set=None,
        data_name=None,
        load_test_data=False,
        enable_rotation=False,
        data_type="bed",
        bed_dataset_path=None,
        bed_dataset_seed=42,
        bed_dataset_shuffle_orders=True,
        bed_target_height=2000,
        max_boxes=300,
        max_candidates=127,
        max_points=0,
        reward_type=None,
        action_scheme="heightmap",
        k_placement=100,
        is_render=False,
        is_hold_on=False,
        **kwags
    ) -> None:
        if load_test_data:
            raise ValueError("현재 브랜치에서는 load_test_data 모드를 지원하지 않습니다.")
        if data_type != "bed":
            raise ValueError("현재 브랜치에서는 BED 모드(data_type='bed')만 지원합니다.")

        self.bin_size = container_size
        self.area = int(self.bin_size[0] * self.bin_size[1])
        self.container = Container(*self.bin_size, rotation=enable_rotation)
        self.can_rotate = enable_rotation
        self.data_type = data_type
        self.reward_type = reward_type
        self.action_scheme = action_scheme
        # 후보 생성 상한과 정책 액션 크기를 분리하지 않고 동일하게 유지한다.
        self.max_candidates = int(max_candidates)
        self.k_placement = int(max_candidates)
        self.max_boxes = int(max_boxes)
        self.max_points = int(max_points)
        self.container.configure_observation_buffer(max_boxes=self.max_boxes)
        self.current_order_id = None
        self.current_order_target = None
        # BED 주문을 모두 소진했을 때 종료 관측을 만들기 위한 플래그다.
        self._bed_no_more_items = False
        self._bed_target_height = int(bed_target_height)
        self._candidate_records: List[Dict[str, object]] = []

        if action_scheme == "EMS":
            self.candidates = np.zeros((self.k_placement, 6), dtype=np.int32)  # (x1, y1, z1, x2, y2, H)
        else:
            self.candidates = np.zeros((self.k_placement, 3), dtype=np.int32)  # (x, y, z)

        dataset_path = bed_dataset_path or data_name
        if dataset_path is None:
            raise ValueError("BED 모드에서는 bed_dataset_path 또는 data_name이 필요합니다.")
        print(f"using BED order dataset: {dataset_path}")
        self.box_creator = BedBoxCreator(
            data_name=dataset_path,
            dataset_seed=bed_dataset_seed,
            dataset_shuffle_orders=bed_dataset_shuffle_orders,
        )
        meta = self.box_creator.start_new_episode()
        self.current_order_id = meta.get("order_id")
        self.current_order_target = meta.get("target")
        self._apply_bed_target_bin(self.current_order_target)

        if is_render:
            self.renderer = VTKRender(container_size, auto_render=not is_hold_on)
        self.render_box = None
        
        self._set_space()

    def _apply_bed_target_bin(self, target):
        # BED target 타입을 실제 팔레트 mm 크기로 매핑한다.
        if target == "euro-pallet":
            self.bin_size = (1200, 800, self._bed_target_height)
        elif target == "rollcontainer":
            self.bin_size = (800, 700, self._bed_target_height)
        else:
            # 알 수 없는 target은 안전하게 euro-pallet 기본값으로 처리한다.
            self.bin_size = (1200, 800, self._bed_target_height)
        self.area = int(self.bin_size[0] * self.bin_size[1])
        self.container = Container(
            *self.bin_size,
            rotation=self.can_rotate,
            max_boxes=self.max_boxes,
        )
        self._set_space()

    def _set_space(self) -> None:
        # 고정 길이 관측을 사용해 bin 크기가 바뀌어도 텐서 shape가 유지되도록 구성한다.
        self.action_space = spaces.Discrete(self.max_candidates)
        obs_dict = {
            "boxes_array": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.max_boxes, 10), dtype=np.float32
            ),
            "next_box": spaces.Box(
                low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.float32
            ),
            "bin_dims": spaces.Box(
                low=1, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "candidate_boxes": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.max_candidates, 3), dtype=np.float32
            ),
            "candidate_positions": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.max_candidates, 3), dtype=np.float32
            ),
            "candidate_oris": spaces.Box(
                low=-1, high=1, shape=(self.max_candidates,), dtype=np.int32
            ),
            "mask": spaces.Box(
                low=0, high=1, shape=(self.max_candidates,), dtype=np.bool_
            ),
        }
        if self.max_points > 0:
            obs_dict["point_cloud"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.max_points, 3), dtype=np.float32
            )
            obs_dict["point_cloud_mask"] = spaces.Box(
                low=0, high=1, shape=(self.max_points,), dtype=np.bool_
            )
        self.observation_space = spaces.Dict(obs_dict)

    def get_box_ratio(self):
        coming_box = self.next_box
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (
                self.container.dimension[0] * self.container.dimension[1] * self.container.dimension[2])

    # box mask (W x L x 3)
    def get_box_plain(self):
        coming_box = self.next_box
        x_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[0]
        y_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[1]
        z_plain = np.ones(self.container.dimension[:2], dtype=np.int32) * coming_box[2]
        return x_plain, y_plain, z_plain

    @property
    def cur_observation(self):
        boxes_array = self._build_boxes_array()
        self._candidate_records = []
        next_box = np.zeros((2, 3), dtype=np.float32)

        if not self._bed_no_more_items:
            size = list(self.next_box)
            next_box[0] = np.asarray([size[0], size[1], size[2]], dtype=np.float32)
            next_box[1] = np.asarray([size[1], size[0], size[2]], dtype=np.float32)
            placements, raw_mask = self.get_possible_position(size)
            self._candidate_records = self._build_candidate_records(size, placements, raw_mask)

        candidate_boxes, candidate_positions, candidate_oris, candidate_mask = self._encode_candidates(
            self._candidate_records
        )

        obs = {
            "boxes_array": boxes_array,
            "next_box": next_box,
            "bin_dims": np.asarray(self.bin_size, dtype=np.float32),
            "candidate_boxes": candidate_boxes,
            "candidate_positions": candidate_positions,
            "candidate_oris": candidate_oris,
            "mask": candidate_mask,
        }
        if self.max_points > 0:
            obs["point_cloud"] = np.zeros((self.max_points, 3), dtype=np.float32)
            obs["point_cloud_mask"] = np.zeros((self.max_points,), dtype=np.bool_)
        return obs

    @property
    def next_box(self) -> list:
        return self.box_creator.preview(1)[0]

    def _peek_current_box_type(self) -> str:
        if hasattr(self.box_creator, "recorder"):
            recorder = getattr(self.box_creator, "recorder", [])
            if recorder:
                item = recorder[-1]
                if isinstance(item, (list, tuple)) and len(item) >= 4:
                    return str(item[3])
        return "unknown"

    def _register_box_type(self, box_type: str) -> int:
        return self.container.register_box_type(box_type)

    def _build_boxes_array(self) -> np.ndarray:
        return self.container.get_boxes_array(normalize=False)

    def _build_candidate_records(self, next_box, candidates, mask_2xk):
        if candidates is None or len(candidates) == 0:
            return []
        records: List[Dict[str, object]] = []
        for idx, cand in enumerate(candidates.tolist()):
            pos = [int(cand[0]), int(cand[1]), int(cand[2])]
            for ori in (0, 1):
                if idx >= mask_2xk.shape[1] or int(mask_2xk[ori, idx]) != 1:
                    continue
                if ori == 0:
                    w, d, h = int(next_box[0]), int(next_box[1]), int(next_box[2])
                else:
                    w, d, h = int(next_box[1]), int(next_box[0]), int(next_box[2])
                records.append(
                    {
                        "src": 0,
                        "idx": int(idx),
                        "pos": pos,
                        "orientation": int(ori),
                        "box_dims": [w, d, h],
                    }
                )

        records.sort(
            key=lambda r: (
                int(r["pos"][2]),
                int(r["pos"][1]),
                int(r["pos"][0]),
                int(r["orientation"]),
                int(r["idx"]),
            )
        )
        return records[:self.max_candidates]

    def _encode_candidates(self, records):
        sizes = np.zeros((self.max_candidates, 3), dtype=np.float32)
        poss = np.zeros((self.max_candidates, 3), dtype=np.float32)
        oris = np.zeros((self.max_candidates,), dtype=np.int32)
        mask = np.zeros((self.max_candidates,), dtype=np.bool_)
        n = min(len(records), self.max_candidates)
        for i in range(n):
            rec = records[i]
            sizes[i] = np.asarray(rec["box_dims"], dtype=np.float32)
            poss[i] = np.asarray(rec["pos"], dtype=np.float32)
            oris[i] = int(rec["orientation"])
            mask[i] = True
        return sizes, poss, oris, mask

    def get_possible_position(self, next_box):
        """
            get possible actions for next box
        Args:
            scheme: the scheme how to generate candidates

        Returns:
            candidate action mask, i.e., the position where the current item should be placed
        """
        if self.action_scheme == "heightmap":
            candidates = self.container.candidate_from_heightmap(next_box, self.k_placement)
            mask = np.zeros((2, self.k_placement), dtype=np.int8)
            valid = min(len(candidates), self.k_placement)
            mask[0, :valid] = 1
            if self.can_rotate:
                mask[1, :valid] = 1
        elif self.action_scheme == "EP":
            candidates, mask = self.container.candidate_from_EP(next_box, self.k_placement)
        elif self.action_scheme == "EMS":
            candidates, mask = self.container.candidate_from_EMS(next_box, self.k_placement)
        elif self.action_scheme == "FC": # full coordinate space
            candidates, mask = self.container.candidate_from_FC(next_box)
        else:
            raise NotImplementedError("action scheme not implemented")

        return candidates, mask 

    def idx2pos(self, idx):
        if idx < 0 or idx >= len(self._candidate_records):
            return [0, 0, 0], 0, [0, 0, 0]

        rec = self._candidate_records[idx]
        pos = list(rec["pos"])
        rot = int(rec["orientation"])
        dim = list(rec["box_dims"])
        self.render_box = [dim, pos]
        return pos, rot, dim

    def step(self, action):
        """

        :param action: action index
        :return: cur_observation
                 reward
                 done, Whether to end boxing (i.e., the current box cannot fit in the bin)
                 info
        """
        action = int(action)
        if action < 0 or action >= self.max_candidates or action >= len(self._candidate_records):
            if self.reward_type == "terminal":
                reward = self.container.get_volume_ratio()
            else:
                reward = 0.0
            done = True
            info = {
                'counter': len(self.container.boxes),
                'ratio': self.container.get_volume_ratio(),
                'invalid_action': True,
                'candidate_count': len(self._candidate_records),
                'order_id': self.current_order_id,
                'target': self.current_order_target,
            }
            return self.cur_observation, reward, done, False, info

        selected = self._candidate_records[action]
        pos, rot, _ = self.idx2pos(action)
        current_box_type = self._peek_current_box_type()
 
        current_box_type_id = self._register_box_type(current_box_type)
        succeeded = self.container.place_box(
            self.next_box,
            pos,
            rot,
            box_type=current_box_type,
            box_type_id=current_box_type_id,
        )
        
        if not succeeded:
            if self.reward_type == "terminal":  # Terminal reward
                reward = self.container.get_volume_ratio()
            else:  # Step-wise/Immediate reward
                reward = 0.0
            done = True
            
            self.render_box = [[0, 0, 0], [0, 0, 0]]
            info = {
                'counter': len(self.container.boxes),
                'ratio': self.container.get_volume_ratio(),
                'selected_src': int(selected["src"]),
                'selected_idx': int(selected["idx"]),
                'selected_orientation': int(selected["orientation"]),
                'order_id': self.current_order_id,
                'target': self.current_order_target,
            }
            return self.cur_observation, reward, done, False, info

        box_ratio = self.get_box_ratio()
        self.box_creator.drop_box()  # remove current box from the list
        try:
            self.box_creator.generate_box_size()  # add a new box to the list
            self._bed_no_more_items = False
        except StopIteration:
            # BED 주문의 마지막 박스를 배치한 직후에는 에피소드를 종료한다.
            self._bed_no_more_items = True
            reward = self.container.get_volume_ratio() if self.reward_type == "terminal" else box_ratio
            done = True
            info = {
                'counter': len(self.container.boxes),
                'ratio': self.container.get_volume_ratio(),
                'order_done': True,
                'selected_src': int(selected["src"]),
                'selected_idx': int(selected["idx"]),
                'selected_orientation': int(selected["orientation"]),
                'order_id': self.current_order_id,
                'target': self.current_order_target,
            }
            return self.cur_observation, reward, done, False, info

        if self.reward_type == "terminal":
            reward = 0.01
        else:
            reward = box_ratio
        done = False
        info = {
            'counter': len(self.container.boxes),
            'ratio': self.container.get_volume_ratio(),
            'selected_src': int(selected["src"]),
            'selected_idx': int(selected["idx"]),
            'selected_orientation': int(selected["orientation"]),
            'order_id': self.current_order_id,
            'target': self.current_order_target,
        }

        return self.cur_observation, reward, done, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        meta = self.box_creator.reset()
        if not isinstance(meta, dict):
            raise RuntimeError("BED creator reset 결과가 비어 있습니다.")
        self.current_order_id = meta.get("order_id")
        self.current_order_target = meta.get("target")
        self._apply_bed_target_bin(self.current_order_target)
        self._bed_no_more_items = False
        self._candidate_records = []
        try:
            self.box_creator.generate_box_size()
        except StopIteration:
            self._bed_no_more_items = True
        self.candidates = np.zeros_like(self.candidates)
        return self.cur_observation, {}
    
    def seed(self, s=None):
        np.random.seed(s)

    def render(self):
        self.renderer.add_item(self.render_box[0], self.render_box[1])
        # self.renderer.save_img()
