import os
import sys
from pathlib import Path
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)  
sys.path.append(parent_path) 

import random 

import gymnasium as gym
import torch
from tianshou.utils.net.common import ActorCritic

from ts_train import build_net
import arguments
from tools import *
from mycollector import PackCollector
from masked_ppo import MaskedPPOPolicy


def _get_env_value(args, key, default=None):
    if key in args.env:
        return args.env[key]
    return default


def _resolve_dataset_path(path_str: str) -> str:
    p = Path(path_str)
    if p.exists():
        return str(p)
    alt = Path(__file__).resolve().parent / p
    if alt.exists():
        return str(alt)
    return str(p)


def _build_eval_env_kwargs(args):
    max_candidates = int(_get_env_value(args, "max_candidates", _get_env_value(args, "k_placement", 80)))
    eval_dataset_path = _get_env_value(
        args,
        "bed_dataset_path_test",
        _get_env_value(
            args,
            "bed_dataset_path_val",
            _get_env_value(args, "bed_eval_dataset_path", "data/bed_bpp/splits/bed_bpp_v1_val.json"),
        ),
    )
    eval_dataset_path = _resolve_dataset_path(str(eval_dataset_path))
    return dict(
        container_size=_get_env_value(args, "container_size", [10, 10, 10]),
        enable_rotation=_get_env_value(args, "rot", True),
        data_type="bed",
        bed_dataset_path=eval_dataset_path,
        bed_dataset_seed=int(_get_env_value(args, "bed_dataset_seed", args.seed)),
        bed_dataset_shuffle_orders=bool(_get_env_value(args, "bed_dataset_shuffle_orders", True)),
        bed_target_height=int(_get_env_value(args, "bed_target_height", 2000)),
        reward_type=args.train.reward_type,
        action_scheme=_get_env_value(args, "scheme", "EMS"),
        max_boxes=int(_get_env_value(args, "max_boxes", 300)),
        max_candidates=max_candidates,
        max_points=int(_get_env_value(args, "max_points", 0)),
        k_placement=max_candidates,
        item_set=[],
        is_render=args.render,
    )


def test(args):

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda", args.device)
    else:
        device = torch.device("cpu")
        
    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # 평가는 val/test split 중 설정된 BED 경로를 사용한다.
    test_env = gym.make(args.env.id, **_build_eval_env_kwargs(args))

    # network
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    
    # RL agent 
    dist = CategoricalMasked

    policy = MaskedPPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.train.gamma,
        eps_clip=args.train.clip_param,
        advantage_normalization=False,
        vf_coef=args.loss.value,
        ent_coef=args.loss.entropy,
        gae_lambda=args.train.gae_lambda,
        action_space=test_env.action_space,
    )
    
    policy.eval()
    try:
        policy.load_state_dict(torch.load(args.ckp, map_location=device))
        # print(policy)
    except FileNotFoundError:
        print("No model found")
        exit()

    test_collector = PackCollector(policy, test_env)

    # Evaluation
    result = test_collector.collect(n_episode=args.test_episode, render=args.render)
    for i in range(args.test_episode):
        order_id = result["order_ids"][i] if i < len(result["order_ids"]) else ""
        target = result["targets"][i] if i < len(result["targets"]) else ""
        done_flag = bool(result["order_done_flags"][i]) if i < len(result["order_done_flags"]) else False
        print(
            f"episode {i+1}\t => \tratio: {result['ratios'][i]:.4f} \t| total: {result['nums'][i]}\t| "
            f"order_done: {int(done_flag)}\t| order_id: {order_id}\t| target: {target}"
        )
    print('All cases have been done!')
    print('----------------------------------------------')
    print("order completion rate: %.4f" % (result.get("order_completion_rate", 0.0)))
    print('average space utilization: %.4f'%(result['ratio']))
    print('average put item number: %.4f'%(result['num']))
    print("standard variance: %.4f"%(result['ratio_std']))


if __name__ == '__main__':
    registration_envs()
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps  

    if args.render:
        args.test_episode = 1  # for visualization

    args.seed = 5
    print(f"dimension: {_get_env_value(args, 'container_size', [10, 10, 10])}")
    test(args)
