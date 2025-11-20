# train.py
from __future__ import annotations

import os
import argparse
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any, Iterable, Union

import yaml
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from graph import *  # expects HeteroGNN, PreferencePredictor
from util import (
    build_config,
    build_hetero_data,
    sample_metadata,
    build_visible_edges,
    parse_jsonl_pairs,  # legacy evaluation: pairs-only
)

# Interaction tuple: (user, session, query, llm, response)
InteractionUnit = Tuple[int, int, int, int, int]

# ──────────────────────────────────────────────────────────────────────────────
# Cached loss objects for grouped training (avoid re-instantiation on each call)
# ──────────────────────────────────────────────────────────────────────────────

_LOSS_CACHE: Dict[torch.device, Tuple[nn.CrossEntropyLoss, nn.BCEWithLogitsLoss]] = {}


def _get_group_losses(device: torch.device) -> Tuple[nn.CrossEntropyLoss, nn.BCEWithLogitsLoss]:
    """
    Return (CrossEntropyLoss, BCEWithLogitsLoss) allocated on the specified device.

    Loss modules are created once per device and reused afterwards. This is
    equivalent in behavior to constructing fresh instances on every call,
    because these loss modules are stateless.
    """
    ce, bce = _LOSS_CACHE.get(device, (None, None))
    if ce is None:
        ce = nn.CrossEntropyLoss().to(device)
        bce = nn.BCEWithLogitsLoss().to(device)
        _LOSS_CACHE[device] = (ce, bce)
    return ce, bce


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def _preconfigure_env_for_determinism(seed: Optional[int]) -> None:
    """
    Set environment variables that influence determinism.

    This function must be invoked before any CUDA tensors are created, otherwise
    some settings (e.g., CUBLAS workspace configuration) may not take effect.
    """
    if seed is None:
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_global_seed(seed: int, cuda_deterministic: bool = True, num_threads: Optional[int] = None) -> None:
    """
    Apply best-effort determinism across Python, NumPy, and PyTorch.

    Note:
        Even with these settings, some GPU operations may remain
        nondeterministic. Use this configuration as a best effort rather
        than a guarantee.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if num_threads is not None:
        try:
            torch.set_num_threads(int(num_threads))
        except Exception:
            pass

    torch.backends.cudnn.benchmark = False
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: pair/group detection
# ──────────────────────────────────────────────────────────────────────────────

def _as_group(item: Any) -> Optional[List[InteractionUnit]]:
    """
    Normalize an item from predict_list to a list of InteractionUnit.

    Accepted forms:
      • (i1, i2) -> [i1, i2]
      • [i1, i2, ..., iK] -> returned as a list

    Returns:
        A list of InteractionUnit if the input can be interpreted as a group,
        otherwise None.
    """
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and all(isinstance(x, tuple) and len(x) == 5 for x in item)
    ):
        return [item[0], item[1]]
    if (
        isinstance(item, (list, tuple))
        and len(item) >= 2
        and all(isinstance(x, tuple) and len(x) == 5 for x in item)
    ):
        return list(item)
    return None


def _all_pairs(predict_list: List[Any]) -> bool:
    """
    Return True if and only if every element of predict_list is exactly a pair.

    This is used to detect when we are in the legacy pairwise setting where
    all items are of the form (InteractionUnit, InteractionUnit).
    """
    for item in predict_list:
        if not (isinstance(item, tuple) and len(item) == 2):
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Loss & Metrics — ORIGINAL PAIRWISE (unchanged behavior)
# ──────────────────────────────────────────────────────────────────────────────

def compute_pairwise_loss_and_metrics(
    predict_list: List[Tuple[InteractionUnit, InteractionUnit]],
    predictor,
    user_emb: torch.Tensor,
    query_emb: torch.Tensor,
    llm_emb: torch.Tensor,
    response_rating: torch.Tensor,
    device: torch.device,
    skip_user_zero: bool = True,
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute the pairwise preference loss and associated metrics.

    This function preserves the original pairwise behavior:
      - It skips ties (pairs for which the two ratings are equal).
      - It can optionally skip pairs involving user_id == 0.
      - It returns:
          * loss: mean pairwise softplus loss over all considered pairs
          * accuracy: fraction of correctly ordered pairs
          * AUC: ROC-AUC over score differences
    """
    pair_losses: List[torch.Tensor] = []
    correct = 0
    total = 0
    score_diffs: List[float] = []
    labels: List[int] = []

    # Cheap aliases
    user_e = user_emb
    query_e = query_emb
    llm_e = llm_emb
    softplus = F.softplus

    # Ground-truth ratings are constants, no gradients needed; keep them on CPU.
    ratings_cpu: List[float] = response_rating.detach().view(-1).cpu().tolist()

    for i1, i2 in predict_list:
        u, _, q, m1, r1 = i1
        u2, _, _, m2, r2 = i2

        assert u == u2, "Each pair must share the same user."
        if skip_user_zero and u == 0:
            continue

        s1 = predictor(user_e[u], query_e[q], llm_e[m1])
        s2 = predictor(user_e[u], query_e[q], llm_e[m2])

        t1 = float(ratings_cpu[r1])
        t2 = float(ratings_cpu[r2])
        if t1 == t2:
            continue  # Skip ties.

        y = 1.0 if t1 > t2 else -1.0
        diff = s1 - s2
        pair_losses.append(softplus(-y * diff))

        total += 1
        if (s1 > s2 and t1 > t2) or (s2 > s1 and t2 > t1):
            correct += 1

        score_diffs.append(float(diff.detach()))
        labels.append(1 if t1 > t2 else 0)

    if pair_losses:
        loss = torch.stack(pair_losses).mean()
    else:
        loss = torch.tensor(0.0, requires_grad=True, device=device)

    acc = (correct / total) if total else 0.0
    try:
        auc = roc_auc_score(labels, score_diffs) if len(set(labels)) > 1 else 0.0
    except ValueError:
        auc = 0.0
    return loss, acc, auc


# ──────────────────────────────────────────────────────────────────────────────
# Loss & Metrics — NEW GROUPED (only when any group K>2)
# ──────────────────────────────────────────────────────────────────────────────

def compute_multigroup_loss_and_metrics(
    predict_list: List[Any],                 # items are pairs or groups
    predictor,                               # (u:[D], q:[D], m:[K,D]) -> [K]
    user_emb: torch.Tensor,                  # [U,D]
    query_emb: torch.Tensor,                 # [Q,D]
    llm_emb: torch.Tensor,                   # [M_all,D]
    response_rating: torch.Tensor,           # [R,1] in [0,1]
    device: torch.device,
    binary: bool = False,                    # yaml train.binary
    skip_user_zero: bool = True,
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute the grouped objective for K-way comparisons (K >= 2).

    Behavior:
      - When binary is False:
          * Use CrossEntropyLoss with a single-label target defined by the
            unique argmax of the rating vector. Groups with ties on the top
            rating are skipped.
      - When binary is True:
          * Use BCEWithLogitsLoss against the rating vector in [0, 1] for
            multi-label style regression to preference scores.

    Metrics:
      - Top-1 accuracy with respect to the argmax of the ground-truth rating.
      - Micro ROC-AUC in a one-vs-rest formulation over the logits.
    """

    ce, bce = _get_group_losses(device)

    losses: List[torch.Tensor] = []
    mc_correct = 0
    mc_total = 0
    micro_scores: List[float] = []
    micro_labels: List[int] = []

    # Aliases for slightly cheaper indexing
    user_e = user_emb
    query_e = query_emb
    llm_e = llm_emb

    for item in predict_list:
        group = _as_group(item)
        if group is None or len(group) < 2:
            continue

        u = group[0][0]
        if skip_user_zero and u == 0:
            continue
        if any(g[0] != u for g in group):
            continue

        q_idx = group[0][2]
        if any(q_idx != g[2] for g in group[1:]):
            continue

        q_vec = query_e[q_idx]
        r_idx = torch.tensor([g[4] for g in group], device=device)
        truth = response_rating[r_idx].squeeze(-1)  # [K] in [0,1]
        m_idx = torch.tensor([g[3] for g in group], device=device)
        M = llm_e[m_idx]                            # [K,D]

        logits = predictor(user_e[u], q_vec, M)     # [K]

        if binary:
            # Fit normalized preference vector directly.
            losses.append(bce(logits, truth))
        else:
            # Single-label objective: target is the unique argmax; skip ties.
            max_val = torch.max(truth)
            gold_idxs = (truth == max_val).nonzero(as_tuple=True)[0]
            if gold_idxs.numel() != 1:
                continue
            gold = int(gold_idxs.item())
            losses.append(ce(logits.unsqueeze(0), torch.tensor([gold], device=device)))

            pred = int(torch.argmax(logits).item())
            mc_total += 1
            if pred == gold:
                mc_correct += 1

            y_bin = torch.zeros_like(truth, dtype=torch.int)
            y_bin[gold] = 1
            micro_labels.extend(y_bin.tolist())
            micro_scores.extend(logits.detach().tolist())

    loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True, device=device)
    acc = (mc_correct / mc_total) if mc_total else 0.0
    try:
        auc = roc_auc_score(micro_labels, micro_scores) if (len(set(micro_labels)) > 1) else 0.0
    except ValueError:
        auc = 0.0
    return loss, acc, auc


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────

def save_if_best(
    ckpt_path: str,
    current_value: float,
    best_value: float,
    minimize: bool,
    epoch: int,
    model,
    predictor,
    emb_dim: int,
    user_dict: Dict[int, str],
    metric: str,
) -> float:
    """
    Save model and predictor parameters when the validation metric improves.

    Args:
        ckpt_path: Target path for the checkpoint file.
        current_value: Current value of the monitored metric.
        best_value: Best value observed so far.
        minimize: If True, lower is better; otherwise, higher is better.
        epoch: Current epoch index.
        model: HeteroGNN model instance.
        predictor: Preference predictor instance.
        emb_dim: Embedding dimension.
        user_dict: Mapping from user index to user identifier.
        metric: Name of the monitored metric (e.g., "auc" or "acc").

    Returns:
        The updated best metric value.
    """
    improved = (current_value < best_value) if minimize else (current_value > best_value)
    if not improved:
        return best_value

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    if not hasattr(model, "num_layers"):
        model.num_layers = None

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "epoch": epoch,
            "metric": metric,
            "score": current_value,
            "config": {"emb_dim": emb_dim, "num_users": len(user_dict), "num_layers": model.num_layers},
        },
        ckpt_path,
    )
    print(f"[Epoch {epoch}] New BEST {metric.upper()}: {current_value:.4f} → saved {ckpt_path}")
    return current_value


def save_periodic(
    ckpt_path: str,
    epoch: int,
    model,
    predictor,
    emb_dim: int,
    user_dict: Dict[int, str],
    metric: Optional[str] = None,
    score: Optional[float] = None,
) -> None:
    """
    Save a periodic checkpoint (one checkpoint file per configured epoch multiple).

    This is intended for debugging, ablation, or analysis across training
    trajectories, independent of the best-validation checkpoint.
    """
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    if not hasattr(model, "num_layers"):
        model.num_layers = None

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "epoch": epoch,
            "metric": metric,
            "score": score,
            "config": {"emb_dim": emb_dim, "num_users": len(user_dict), "num_layers": model.num_layers},
        },
        ckpt_path,
    )
    print(f"[Epoch {epoch}] Saved periodic checkpoint → {ckpt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# JSONL grouping helper (for evaluation)
# ──────────────────────────────────────────────────────────────────────────────

def _iter_groups_from_jsonl(path: str):
    """
    Stream groups of consecutive JSONL records that share the same key
    (question_id, turn, judge).

    This reproduces the grouping logic used in the original evaluation helper:
    each yielded group corresponds to one question, turn, and judge combination.
    """
    import json

    key = None
    block = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            k = (e["question_id"], e["turn"], e["judge"])
            if key is None or k == key:
                block.append(e)
                key = k
            else:
                if block:
                    yield block
                block = [e]
                key = k
    if block:
        yield block


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation on JSONL (pairs or multiway). Also accepts pre-parsed groups.
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_jsonl(
    jsonl_or_groups: Union[str, Iterable[List[dict]]],
    predictor,
    agg_emb: Dict[str, torch.Tensor],
    user_dict: Dict[int, str],
    llm_dict: Dict[int, str],
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate the predictor on JSONL-formatted data or pre-parsed groups.

    The function supports:
      - Pairwise groups (K = 2, legacy behavior).
      - Multiway groups (K > 2), where all models are scored jointly.

    Args:
        jsonl_or_groups:
            Either:
              * a string path to a JSONL file, which will be streamed and grouped
                on the fly, or
              * an iterable of groups, where each group is a list of dictionaries
                in the same format as produced by _iter_groups_from_jsonl.
        predictor:
            A callable with signature:
                - K=2 (legacy): predictor(u, q, m) where m is a single model embedding.
                - K>2: predictor(u, q, M) where M is a [K, D] tensor of model embeddings.
        agg_emb:
            Dictionary containing aggregated node embeddings, at least
            {"user": ..., "llm": ...}.
        user_dict:
            Mapping from user index to user identifier (judge).
        llm_dict:
            Mapping from LLM index to model name.
        device:
            Target device for tensors used during evaluation.
        verbose:
            If True, prints aggregate evaluation statistics.

    Returns:
        A dictionary containing:
            - "acc": overall accuracy
            - "auc": overall ROC-AUC
            - "pairs_seen": number of grouped conversations processed
            - "eval_pairs": number of turn-level decisions
            - "per_llm": per-model (correct, total) counts
            - "per_user": per-user (correct, total) counts
            - "per_user_auc": per-user ROC-AUC in the pairwise setting
            - "pred_win": per-model (wins, total_predictions) statistics
    """

    if isinstance(jsonl_or_groups, str):
        groups_iter = _iter_groups_from_jsonl(jsonl_or_groups)
    else:
        groups_iter = iter(jsonl_or_groups)

    user_name2idx = {name: idx for idx, name in user_dict.items()}
    llm_name2idx = {name: idx for idx, name in llm_dict.items()}

    total, correct, groups_seen = 0, 0, 0
    preds_for_auc, labels_for_auc = [], []

    llm_correct = defaultdict(int)
    llm_total   = defaultdict(int)
    llm_pred_win   = defaultdict(int)
    llm_pred_total = defaultdict(int)
    user_correct = defaultdict(int)
    user_total   = defaultdict(int)
    user_preds   = defaultdict(list)  # Only populated in the K=2 path (legacy per-user AUC).
    user_labels  = defaultdict(list)

    all_groups_are_pairs = True  # Used to detect pure legacy behavior when applicable.

    # Aliases for embeddings
    user_emb = agg_emb["user"]
    llm_emb = agg_emb["llm"]
    emb_dtype = user_emb.dtype

    for group in groups_iter:
        groups_seen += 1
        K = len(group)
        if K != 2:
            all_groups_are_pairs = False

        # Resolve user once per group (all entries share the same judge by construction).
        uname = group[0]["judge"]
        uid = user_name2idx.get(uname)
        if uid is None:
            continue
        uvec = user_emb[uid]

        # Collect conversations for this group.
        conv_lists = [g["conversation"] for g in group]
        # Iterate turns synchronously across models.
        for turn_items in zip(*conv_lists):
            # Ensure all queries at this turn are identical.
            q_embs = [ti["query_emb"] for ti in turn_items]
            if any(q_embs[0] != q for q in q_embs[1:]):
                continue  # Skip inconsistent rows.
            q = torch.tensor(q_embs[0], device=device, dtype=emb_dtype)

            # Ratings (assumed to have been normalized upstream).
            ratings = [float(ti["rating"]) for ti in turn_items]

            # LLM indices and embeddings.
            model_names = [g["model"] for g in group]
            mids = [llm_name2idx.get(mn) for mn in model_names]
            if any(mid is None for mid in mids):
                continue
            M = llm_emb[torch.tensor(mids, device=device)]

            if K == 2:
                # ---------- Legacy pairwise path (unchanged behavior) ----------
                # Call predictor separately for each model to exactly preserve
                # prior math and logging.
                s1 = predictor(uvec, q, M[0]).item()
                s2 = predictor(uvec, q, M[1]).item()

                y1_true, y2_true = ratings[0], ratings[1]
                if y1_true == y2_true:
                    continue

                # Score difference for AUC (legacy definition).
                preds_for_auc.append(s1 - s2)
                labels_for_auc.append(1 if y1_true > y2_true else 0)

                total += 1
                user_total[uid] += 1
                llm_total[mids[0]] += 1
                llm_total[mids[1]] += 1
                llm_pred_total[mids[0]] += 1
                llm_pred_total[mids[1]] += 1

                # Predicted winner (for "pred_win" statistics).
                if s1 > s2:
                    llm_pred_win[mids[0]] += 1
                else:
                    llm_pred_win[mids[1]] += 1

                # Correctness: both models are credited on a correct decision.
                if (s1 > s2 and y1_true > y2_true) or (s2 > s1 and y2_true > y1_true):
                    correct += 1
                    user_correct[uid] += 1
                    llm_correct[mids[0]] += 1
                    llm_correct[mids[1]] += 1

                # Per-user AUC (legacy, based on score differences).
                user_preds[uid].append(s1 - s2)
                user_labels[uid].append(1 if y1_true > y2_true else 0)

            else:
                # ---------- Multiway path (K>2): jointly score all models ----------
                logits = predictor(uvec, q, M)  # [K]; supports batched model embeddings.
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                # Skip groups with ties on the top rating to mirror training behavior.
                truth = torch.tensor(ratings, device=device)
                max_val = torch.max(truth)
                gold_idxs = (truth == max_val).nonzero(as_tuple=True)[0]
                if gold_idxs.numel() != 1:
                    continue
                gold = int(gold_idxs.item())
                pred = int(torch.argmax(logits).item())

                total += 1
                user_total[uid] += 1

                # Per-LLM counts: each involved model contributes to total counts.
                for mid in mids:
                    llm_total[mid] += 1
                    llm_pred_total[mid] += 1
                llm_pred_win[mids[pred]] += 1

                if pred == gold:
                    correct += 1
                    user_correct[uid] += 1
                    # As in the pairwise path, credit all involved models
                    # for a correct decision at the group level.
                    for mid in mids:
                        llm_correct[mid] += 1

                # Micro AUC in a one-vs-rest formulation using the logits.
                for k in range(K):
                    labels_for_auc.append(1 if k == gold else 0)
                    preds_for_auc.append(float(logits[k].item()))

    # Aggregate global metrics.
    acc = correct / total if total else 0.0
    try:
        auc = roc_auc_score(labels_for_auc, preds_for_auc) if (len(set(labels_for_auc)) > 1) else 0.0
    except ValueError:
        auc = 0.0

    if verbose:
        mode = "pairs" if all_groups_are_pairs else "multi"
        print(f"[Eval/{mode}] groups={groups_seen} | decisions={total} | acc={acc:.3f} | auc={auc:.3f}")

    # Per-user AUC is only defined in the legacy pairwise setting.
    per_user_auc: Dict[str, float] = {}
    for uid, lbls in user_labels.items():
        try:
            per_user_auc[user_dict[uid]] = roc_auc_score(lbls, user_preds[uid]) if len(set(lbls)) > 1 else 0.0
        except ValueError:
            per_user_auc[user_dict[uid]] = 0.0

    return {
        "acc": acc,
        "auc": auc,
        "pairs_seen": groups_seen,  # Number of grouped conversations processed.
        "eval_pairs": total,        # Number of turn-level decisions evaluated.
        "per_llm": {llm_dict[mid]: (llm_correct[mid], llm_total[mid]) for mid in llm_total},
        "per_user": {user_dict[uid]: (user_correct[uid], user_total[uid]) for uid in user_total},
        "per_user_auc": per_user_auc,
        "pred_win": {llm_dict[mid]: (llm_pred_win[mid], llm_pred_total[mid]) for mid in llm_pred_total},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_with_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train the HeteroGNN preference model using a YAML-style configuration.

    The configuration dictionary is expected to contain:
      - "dataset": dataset-specific paths and identifiers.
      - "train": training hyperparameters (e.g., epochs, learning rate,
                 record_per_user, multi_turn, binary, objective).
      - "checkpoint" (optional): checkpointing behavior (e.g., save_every, root).

    Returns:
        A dictionary containing the trained model, predictor, graph, and
        associated metadata and paths.
    """
    train_cfg = cfg.get("train", {})
    dataset_cfg = cfg["dataset"]
    checkpoint_cfg = cfg.get("checkpoint", {})

    # Determinism (environment first, then seeds) – must be done before any CUDA tensor is created.
    seed = int(train_cfg.get("seed", 0))
    _preconfigure_env_for_determinism(seed)
    set_global_seed(seed, cuda_deterministic=True)

    # Data locations.
    base_dir = f"{dataset_cfg['path'].rstrip('/')}/{dataset_cfg['name']}"
    train_jsonl = f"{base_dir}/training_set.jsonl"
    train_pt = f"{base_dir}/training_set.pt"

    # Frequency of validation runs (1 = every epoch; default preserves original behavior).
    eval_every = int(train_cfg.get("eval_every", 1))

    # Build or load graph tensors and associated metadata.
    if cfg.get("config", {}).get("preprocess", False):
        res = build_config(train_pt)
    else:
        res = build_config(train_jsonl, ckpt_path=train_pt)

    (config, nodes, edges, user_dict, _qdict, _rdict,
     response_rating, llm_dict, metadata, device) = res

    if "aggregation_type" in train_cfg:
        config.aggregation_type = train_cfg["aggregation_type"]
    else:
        config.aggregation_type = "mean"

    # Model initialization.
    dim = config.emb_dim
    graph_data = build_hetero_data(nodes, edges)
    num_layers = 3 if train_cfg.get("multi_turn", False) else 2

    model = HeteroGNN(graph_data.metadata(), num_users=len(user_dict), emb_dim=dim, num_layers=num_layers).to(device)
    model.num_layers = num_layers

    predictor = PreferencePredictor(dim=dim).to(device)
    optim = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=float(train_cfg.get("lr", 5e-4)),
    )

    # Sampling and training-loop settings.
    min_per_user = int(train_cfg.get("record_per_user", 10))
    predict_count = int(train_cfg.get("prediction_count", 256))
    multi_turn = bool(train_cfg.get("multi_turn", False))
    use_binary = bool(train_cfg.get("binary", False))

    metric = str(train_cfg.get("objective", "auc")).lower()  # "auc" | "acc"
    minimize = False
    best = float("-inf")

    # Checkpoint configuration.
    run_id = train_cfg.get("id", "default")
    ckpt_root = checkpoint_cfg.get("root", "./models")
    save_every = int(checkpoint_cfg.get("save_every", 25))

    ckpt_dir = os.path.join(ckpt_root, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_ckpt_path = os.path.join(ckpt_dir, f"{dataset_cfg['name']}_best_model.pt")

    # Validation setup: prefer JSONL if available, and parse once for reuse.
    val_jsonl = os.path.join(base_dir, "valid_set.jsonl")
    val_groups: Optional[List[List[dict]]] = None
    if os.path.isfile(val_jsonl):
        # Parse once and reuse across epochs; semantics match streaming evaluation.
        val_groups = list(_iter_groups_from_jsonl(val_jsonl))

    # Main epoch loop.
    epochs = int(train_cfg.get("epochs", 350))
    for epoch in range(1, epochs + 1):
        model.train()
        predictor.train()
        optim.zero_grad()

        visible_list, predict_list = sample_metadata(
            metadata,
            visible_count=min_per_user * config.num_users,
            predict_count=predict_count,
            min_record_per_user=min_per_user,
            seed=(seed or 0) + epoch,
        )

        visible_edges = build_visible_edges(edges, visible_list, multi_turn, device)

        # Forward pass.
        agg_emb = model(graph_data.x_dict, visible_edges)
        user_emb, query_emb, llm_emb = agg_emb["user"], agg_emb["query"], agg_emb["llm"]

        # If all items are pairs, reproduce legacy logs exactly; otherwise use grouped objective.
        if _all_pairs(predict_list):
            loss, acc, auc = compute_pairwise_loss_and_metrics(
                predict_list, predictor, user_emb, query_emb, llm_emb, response_rating, device
            )
        else:
            loss, acc, auc = compute_multigroup_loss_and_metrics(
                predict_list, predictor, user_emb, query_emb, llm_emb, response_rating, device, binary=use_binary
            )

        loss.backward()
        optim.step()

        # Validation (if groups are available).
        model.eval()
        predictor.eval()
        current_val = None
        if val_groups is not None and (epoch % eval_every == 0 or epoch == epochs):
            val = evaluate_jsonl(val_groups, predictor, agg_emb, user_dict, llm_dict, device, verbose=False)
            current_val = val.get(metric, 0.0)
            best = save_if_best(
                best_ckpt_path,
                current_val,
                best,
                minimize,
                epoch,
                model,
                predictor,
                dim,
                user_dict,
                metric,
            )

        # Logging and periodic checkpointing (rolling by epoch).
        if save_every and (epoch % save_every == 0):
            msg = f"[Epoch {epoch}] loss={loss.item():.4f} acc={acc:.3f} auc={auc:.3f}"
            if current_val is not None:
                msg += f" | val_{metric}={current_val:.3f}"
            print(msg)

            periodic_path = os.path.join(ckpt_dir, f"{dataset_cfg['name']}_ckpt_{epoch}.pt")
            save_periodic(
                periodic_path,
                epoch,
                model,
                predictor,
                dim,
                user_dict,
                metric=metric,
                score=current_val,
            )

    # Load the best model before final testing.
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        predictor.load_state_dict(ckpt["predictor_state_dict"])
        print(f"[FINAL] Loaded best epoch {ckpt.get('epoch', '?')} {metric.upper()}={ckpt.get('score', 0):.4f}")

    # Final test: prefer preprocessed test_set.pt; otherwise fall back to test_set.jsonl.
    test_jsonl = os.path.join(base_dir, "test_set.jsonl")
    test_pt = os.path.join(base_dir, "test_set.pt")

    print("\n[TEST] Running final evaluation...")
    if os.path.isfile(test_pt):
        (t_cfg, t_nodes, t_edges, _t_user, _t_q, _t_r, t_ratings, _t_llm, t_meta, t_device) = build_config(test_pt)
        t_graph = build_hetero_data(t_nodes, t_edges)
        with torch.no_grad():
            t_agg = model(t_graph.x_dict, t_edges)
        # Stable iteration order for reproducibility.
        t_pairs = sorted(list(t_meta))
        _t_loss, t_acc, t_auc = compute_pairwise_loss_and_metrics(
            t_pairs, predictor, t_agg["user"], t_agg["query"], t_agg["llm"], t_ratings, t_device
        )
        print(f"[TEST-pt] acc={t_acc:.4f} | auc={t_auc:.4f}")
    elif os.path.isfile(test_jsonl):
        with torch.no_grad():
            full_train_emb = model(graph_data.x_dict, edges)
        test_metrics = evaluate_jsonl(test_jsonl, predictor, full_train_emb, user_dict, llm_dict, device, verbose=True)
        print(f"[TEST-jsonl] acc={test_metrics['acc']:.4f} | auc={test_metrics['auc']:.4f}")
    else:
        print("[TEST] No test_set.jsonl or test_set.pt found; skipping.")

    return {
        "model": model,
        "predictor": predictor,
        "graph": graph_data,
        "user_dict": user_dict,
        "llm_dict": llm_dict,
        "metadata": metadata,
        "device": device,
        "best_ckpt_path": best_ckpt_path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Parse the command-line configuration path and launch training.
    """
    parser = argparse.ArgumentParser(description="Train HeteroGNN preference model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_with_cfg(cfg)


if __name__ == "__main__":
    main()
