import numpy as np
from typing import List, Dict
import random

def load_trajectories_npz(path: str):
    d = np.load(path, allow_pickle=True)
    if 'trajectories' in d:
        return list(d['trajectories'])
    # fallback: expect flattened arrays + traj_idx
    if all(k in d for k in ('states','actions','rewards','traj_idx')):
        states = d['states']
        actions = d['actions']
        rewards = d['rewards']
        traj_idx = d['traj_idx']
        trajectories = []
        unique_ids = np.unique(traj_idx)
        for tid in unique_ids:
            mask = traj_idx == tid
            trajectories.append({
                'states': states[mask],
                'actions': actions[mask],
                'rewards': rewards[mask],
                'dones': np.zeros(mask.sum(), dtype=bool)
            })
            trajectories[-1]['dones'][-1] = True
        return trajectories
    raise ValueError("Unexpected data format. See docstring of data_loader.")

def build_synthetic_demo(num_traj=100, max_T=20, n_features=379, seed=0):
    """
    Build a small synthetic dataset for testing pipeline end-to-end.
    Not intended to reflect real clinical data.
    """
    random = __import__('random')
    random.seed(seed)
    np.random.seed(seed)
    trajs = []
    for _ in range(num_traj):
        T = random.randint(5, max_T)
        s = np.random.randn(T, n_features).astype(np.float32)
        # random actions 0..4
        a = np.random.randint(0,5,size=(T,), dtype=np.int64)
        # reward: zero for intermediate steps, final reward is +1 for survival with prob depending on actions
        final_surv = (a.mean() < 2.5) and (np.random.rand() > 0.3)  # synthetic bias
        r = np.zeros(T, dtype=np.float32)
        r[-1] = 1.0 if final_surv else 0.0
        dones = np.zeros(T, dtype=bool); dones[-1]=True
        trajs.append({'states':s,'actions':a,'rewards':r,'dones':dones})
    return trajs

def flatten_trajectories(trajs: List[Dict]):
    """
    Flatten to arrays and produce traj_idx mapping.
    """
    states = []
    actions = []
    rewards = []
    traj_idx = []
    for i,t in enumerate(trajs):
        states.append(t['states'])
        actions.append(t['actions'])
        rewards.append(t['rewards'])
        traj_idx.append(np.full(len(t['actions']), i, dtype=np.int32))
    return {
        'states': np.concatenate(states, axis=0),
        'actions': np.concatenate(actions, axis=0),
        'rewards': np.concatenate(rewards, axis=0),
        'traj_idx': np.concatenate(traj_idx, axis=0)
    }

if __name__ == "__main__":
    demo = build_synthetic_demo()
    print(len(demo), demo[0]['states'].shape)
