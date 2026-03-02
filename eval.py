import os
import sys
import types
import numpy as np
import torch
from omegaconf import OmegaConf
from tianshou.data import Batch

curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_path)

# render.py 가 최상단에서 `import vtk` 를 하므로, VTK 미설치 환경에서도
# 모듈 import 가 가능하도록 mock 처리 (평가 시 렌더링 불필요)
try:
    import vtk
except (ImportError, ModuleNotFoundError):
    sys.modules["vtk"] = types.ModuleType("vtk")

from envs.Packing.env import PackingEnv
from ts_train import build_net

# --------------- 설정 ---------------
CHECKPOINT_DIR = os.path.join(
    curr_path, "logs",
    "OnlinePack-v1_10-10-10_EMS_80_random_PPO_seed5_Adam_2026.02.13-10-55-16"
)
MODEL_FILE = "policy_step_best.pth"
NUM_EPISODES = 1000
GREEDY = True       # True: argmax, False: stochastic sampling


def select_action(actor, obs_np: dict, device, greedy: bool = True) -> int:
    """obs_np: {"obs": np.ndarray, "mask": np.ndarray}"""
    batch = Batch(obs=obs_np["obs"][np.newaxis, :], mask=obs_np["mask"][np.newaxis, :])
    with torch.no_grad():
        logits, _ = actor(batch, state=None)          # (1, k_placement)
    logits = logits.squeeze(0)

    # 유효하지 않은 액션 마스킹
    mask = torch.as_tensor(obs_np["mask"], dtype=torch.bool, device=device)
    logits = torch.where(mask, logits, torch.tensor(-1e18, device=device))

    if greedy:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.distributions.Categorical(probs).sample().item())


def main():
    # ---------- 설정 로드 ----------
    config_path = os.path.join(CHECKPOINT_DIR, "config.yaml")
    cfg = OmegaConf.load(config_path)

    # arguments.py 와 동일한 방식으로 box_size_set 구성 (RS dataset: 1~5)
    box_small = int(max(cfg.env.container_size) / 10)   # 10/10 = 1
    box_big   = int(max(cfg.env.container_size) / 2)    # 10/2  = 5
    step = cfg.env.get("step") or box_small              # step = 1
    box_size_set = [
        (i, j, k)
        for i in range(box_small, box_big + 1, step)
        for j in range(box_small, box_big + 1, step)
        for k in range(box_small, box_big + 1, step)
    ]
    cfg.env.box_small    = box_small
    cfg.env.box_big      = box_big
    cfg.env.box_size_set = box_size_set

    device = (
        torch.device("cuda", 0) if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device        : {device}")
    print(f"Model         : {MODEL_FILE}")
    print(f"Box size range: {box_small}~{box_big} (step={step}), {len(box_size_set)} combos")
    print(f"Episodes      : {NUM_EPISODES}")
    print(f"Greedy        : {GREEDY}")

    # ---------- 환경 (gym.make 대신 직접 생성 → gymnasium wrapper 우회) ----------
    env = PackingEnv(
        container_size  = cfg.env.container_size,
        enable_rotation = cfg.env.rot,
        data_type       = cfg.env.box_type,
        item_set        = cfg.env.box_size_set,
        reward_type     = cfg.train.reward_type,
        action_scheme   = cfg.env.scheme,
        k_placement     = cfg.env.k_placement,
    )

    # ---------- 모델 ----------
    actor, _ = build_net(cfg, device)

    model_path = os.path.join(CHECKPOINT_DIR, MODEL_FILE)
    policy_state = torch.load(model_path, map_location=device)
    actor_state  = {k[len("actor."):]: v for k, v in policy_state.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_state)
    actor.eval()
    print(f"Loaded        : {model_path}\n")

    # ---------- 평가 ----------
    SEP = "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+"
    print(SEP)
    print(f"| {'Episode':>8} | {'Ratio':>8} | {'# Items':>8} |")
    print(SEP)

    util_hist, num_hist = [], []

    for ep in range(1, NUM_EPISODES + 1):
        # 에피소드별 시드 고정 → RS dataset (재현 가능한 랜덤 시퀀스)
        np.random.seed(ep)
        obs, _ = env.reset()
        done = False

        while not done:
            action = select_action(actor, obs, device, greedy=GREEDY)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        ratio = info["ratio"]
        num   = info["counter"]
        util_hist.append(ratio)
        num_hist.append(num)
        print(f"| {ep:>8} | {ratio:>8.4f} | {num:>8} |")

    print(SEP)

    avg_util = np.mean(util_hist)
    std_util = np.std(util_hist)
    avg_num  = np.mean(num_hist)
    print(f"\nAverage space utilization : {avg_util:.4f}")
    print(f"Standard deviation        : {std_util:.4f}")
    print(f"Average # items packed    : {avg_num:.2f}")


if __name__ == "__main__":
    main()
