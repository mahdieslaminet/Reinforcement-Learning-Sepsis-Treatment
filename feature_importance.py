import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch

def clinician_rf_importance(trajs):
    X = np.concatenate([t['states'] for t in trajs], axis=0)
    y = np.concatenate([t['actions'] for t in trajs], axis=0)
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rf.fit(X,y)
    importances = rf.feature_importances_
    return rf, importances

def permutation_importance_agent(agent, trajs, metric='avg_prob', n_repeats=10):
    """
    For each feature i:
      - compute baseline metric (average prob of agent's chosen action or mean value)
      - permute feature i across all states (shuffle column i) n_repeats times and compute metric drop.
    metric: 'avg_prob' -> average probability agent assigns to actions actually taken in dataset
    """
    # gather dataset
    states = np.concatenate([t['states'] for t in trajs], axis=0)
    actions = np.concatenate([t['actions'] for t in trajs], axis=0)
    agent.model.eval()
    with torch.no_grad():
        s = torch.tensor(states, dtype=torch.float32, device=agent.device)
        logits, values = agent.model(s)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    baseline = np.mean([probs[i, actions[i]] for i in range(len(actions))])
    n_features = states.shape[1]
    importances = np.zeros(n_features)
    for feat in range(n_features):
        scores = []
        for _ in range(n_repeats):
            s_perm = states.copy()
            np.random.shuffle(s_perm[:, feat])
            with torch.no_grad():
                s_t = torch.tensor(s_perm, dtype=torch.float32, device=agent.device)
                logits_p, _ = agent.model(s_t)
                probs_p = torch.softmax(logits_p, dim=-1).cpu().numpy()
            score = np.mean([probs_p[i, actions[i]] for i in range(len(actions))])
            scores.append(baseline - score)
        importances[feat] = np.mean(scores)
    return importances
