from __future__ import annotations

import os
import argparse
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any

import yaml
import random
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T

from graph import *
from util import (
    build_config,
    build_hetero_data,
    sample_metadata,
    build_visible_edges,
    parse_jsonl_pairs,
)

# Interaction index layout: (user, session, query, llm, response)
InteractionUnit = Tuple[int, int, int, int, int]


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def _preconfigure_env_for_determinism(seed: Optional[int]) -> None:
    """
    Configure environment variables that influence determinism.

    This function must be invoked before any CUDA tensor is created to ensure
    that libraries such as cuBLAS and Python hashing behave deterministically
    when a seed is provided.
    """
    if seed is None:
        return
    # This is used by certain CUDA BLAS routines and must be set before the
    # first CUDA context is created.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    # Ensure deterministic dictionary and hash-based iteration order across processes.
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_global_seed(seed: int, cuda_deterministic: bool = True, num_threads: Optional[int] = None) -> None:
    """
    Make a best-effort attempt to enforce determinism across Python, NumPy, and PyTorch.

    Note:
        Full determinism is not guaranteed on all hardware and for all operators.
        Some GPU kernels may remain non-deterministic in practice.
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

    # cuDNN / kernel algorithm choices
    torch.backends.cudnn.benchmark = False
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Add Reverse Edges to Dict
# ──────────────────────────────────────────────────────────────────────────────

def add_reverse_edges_to_dict(edge_index_dict: Dict, device: torch.device) -> Dict:
    """
    Given a dictionary of edge indices, add an explicit reverse edge type and
    edge_index tensor for each existing edge type.

    The new reverse relation is named by prefixing the original relation with
    ``rev_`` and swapping the source and destination node types.
    """
    new_edges = {}
    for edge_type, index in edge_index_dict.items():
        src, rel, dst = edge_type

        new_edges[edge_type] = index

        rev_edge_type = (dst, f"rev_{rel}", src)

        new_edges[rev_edge_type] = index.flip(0)

    return new_edges


# ──────────────────────────────────────────────────────────────────────────────
# Loss & Metrics
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
    Compute pairwise preference loss and associated metrics (accuracy and AUC).

    The function:
        * Uses a pairwise logistic (softplus) loss on score differences.
        * Skips ties where the two responses have equal ratings.
        * Optionally skips pairs for user_id == 0 (to reproduce original behavior).
    """
    pair_losses = []
    correct, total = 0, 0
    score_diffs, labels = [], []

    for i1, i2 in predict_list:
        u, _, q, m1, r1 = i1
        u2, _, _, m2, r2 = i2
        assert u == u2, "Each pair must share the same user."
        if skip_user_zero and u == 0:
            continue

        s1 = predictor(user_emb[u], query_emb[q], llm_emb[m1])
        s2 = predictor(user_emb[u], query_emb[q], llm_emb[m2])

        t1 = float(response_rating[r1])
        t2 = float(response_rating[r2])
        if t1 == t2:
            continue  # skip ties

        y = 1.0 if t1 > t2 else -1.0
        diff = s1 - s2
        pair_losses.append(F.softplus(-y * diff))

        total += 1
        if (s1 > s2 and t1 > t2) or (s2 > s1 and t2 > t1):
            correct += 1

        score_diffs.append(float(diff.detach()))
        labels.append(1 if t1 > t2 else 0)

    loss = torch.stack(pair_losses).mean() if pair_losses else torch.tensor(0.0, requires_grad=True, device=device)
    acc = (correct / total) if total else 0.0
    try:
        auc = roc_auc_score(labels, score_diffs) if len(set(labels)) > 1 else 0.0
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
    Save the model and predictor parameters when the validation metric improves.

    Args:
        ckpt_path: Target path for the checkpoint.
        current_value: Current value of the monitored metric.
        best_value: Best metric value observed so far.
        minimize: If True, lower values are considered better; otherwise higher values are better.
        epoch: Current training epoch.
        model: GNN model instance.
        predictor: Preference prediction head.
        emb_dim: Embedding dimensionality.
        user_dict: Mapping from user indices to user identifiers.
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
    Save a periodic checkpoint.

    This function writes a checkpoint at a fixed interval (for example,
    every N epochs), regardless of whether the validation metric improved.
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
# Evaluation on JSONL (validation/test)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_jsonl(
    jsonl_path: str,
    predictor,
    agg_emb: Dict[str, torch.Tensor],
    user_dict: Dict[int, str],
    llm_dict: Dict[int, str],
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Perform pairwise evaluation on a JSONL file containing model comparisons.

    The evaluation assumes that each pair of entries corresponds to two
    competing model responses for the same user and query, and uses the
    fixed embeddings in ``agg_emb`` to compute preference scores.
    """
    user_name2idx = {name: idx for idx, name in user_dict.items()}
    llm_name2idx = {name: idx for idx, name in llm_dict.items()}

    total, correct, cnt = 0, 0, 0
    preds, labels = [], []

    llm_correct = defaultdict(int)
    llm_total = defaultdict(int)
    user_correct = defaultdict(int)
    user_total = defaultdict(int)
    llm_pred_win = defaultdict(int)
    llm_pred_total = defaultdict(int)
    user_preds = defaultdict(list)
    user_labels = defaultdict(list)

    for e1, e2 in parse_jsonl_pairs(jsonl_path):
        cnt += 1
        assert e1["question_id"] == e2["question_id"]
        assert e1["turn"] == e2["turn"]

        uname = e1["judge"]
        assert uname == e2["judge"]
        uid = user_name2idx.get(uname)
        if uid is None:
            continue

        mname1, mname2 = e1["model"], e2["model"]
        mid1, mid2 = llm_name2idx.get(mname1), llm_name2idx.get(mname2)
        if mid1 is None or mid2 is None:
            continue

        u = agg_emb["user"][uid]
        m1 = agg_emb["llm"][mid1]
        m2 = agg_emb["llm"][mid2]

        for t1, t2 in zip(e1["conversation"], e2["conversation"]):
            q1, q2 = t1["query_emb"], t2["query_emb"]
            if q1 != q2:
                continue
            q = torch.tensor(q1, device=device, dtype=u.dtype)

            y1_true, y2_true = float(t1["rating"]), float(t2["rating"])
            if y1_true == y2_true:
                continue

            y1_pred = predictor(u, q, m1).item()
            y2_pred = predictor(u, q, m2).item()

            preds.append(y1_pred - y2_pred)
            labels.append(1 if y1_true > y2_true else 0)

            total += 1
            user_total[uid] += 1
            llm_total[mid1] += 1
            llm_total[mid2] += 1
            llm_pred_total[mid1] += 1
            llm_pred_total[mid2] += 1

            if y1_pred > y2_pred:
                llm_pred_win[mid1] += 1
            else:
                llm_pred_win[mid2] += 1

            if (y1_true > y2_true and y1_pred > y2_pred) or (y2_true > y1_true and y2_pred > y1_pred):
                correct += 1
                user_correct[uid] += 1
                llm_correct[mid1] += 1
                llm_correct[mid2] += 1

            user_preds[uid].append(y1_pred - y2_pred)
            user_labels[uid].append(1 if y1_true > y2_true else 0)

    acc = correct / total if total else 0.0
    try:
        auc = roc_auc_score(labels, preds) if total else 0.0
    except ValueError:
        auc = 0.0

    if verbose:
        print(f"[Eval] pairs_read={cnt} | decisions={total} | acc={acc:.3f} | auc={auc:.3f}")

    per_user_auc = {}
    for uid, lbls in user_labels.items():
        try:
            per_user_auc[user_dict[uid]] = roc_auc_score(lbls, user_preds[uid]) if len(set(lbls)) > 1 else 0.0
        except ValueError:
            per_user_auc[user_dict[uid]] = 0.0

    return {
        "acc": acc,
        "auc": auc,
        "pairs_seen": cnt,
        "eval_pairs": total,
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
    Train a Heterogeneous GNN preference model using the provided configuration.

    The configuration dictionary is typically loaded from a YAML file and
    includes dataset paths, model hyperparameters, checkpoint settings, and
    training options.
    """
    # Determinism (environment first, then seeds); this must be executed before
    # any CUDA tensor is created.
    seed = int(cfg.get("train", {}).get("seed", 0))
    _preconfigure_env_for_determinism(seed)
    set_global_seed(seed, cuda_deterministic=True)

    # Dataset locations
    base_dir = f"{cfg['dataset']['path'].rstrip('/')}/{cfg['dataset']['name']}"
    train_jsonl = f"{base_dir}/train.jsonl"
    train_pt = f"{base_dir}/train.pt"

    # Build or load graph tensors and metadata
    if cfg.get("config", {}).get("preprocess", False):
        res = build_config(train_pt)
    else:
        res = build_config(train_jsonl, ckpt_path=train_pt)

    (config, nodes, edges, user_dict, _qdict, _rdict,
     response_rating, llm_dict, metadata, device) = res

    if "aggregation_type" in cfg.get("train", {}):
        config.aggregation_type = cfg["train"]["aggregation_type"]
    else:
        config.aggregation_type = "mean"

    # Model initialization
    dim = config.emb_dim
    graph_data = build_hetero_data(nodes, edges)

    graph_data = T.ToUndirected()(graph_data)
    graph_data = graph_data.to(device)

    num_layers = 3 if cfg["train"].get("multi_turn", False) else 2

    model = HeteroGNN(graph_data.metadata(), num_users=len(user_dict), emb_dim=dim, num_layers=num_layers).to(device)
    model.num_layers = num_layers

    predictor = PreferencePredictor(dim=dim).to(device)
    optim = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=float(cfg["train"].get("lr", 5e-4)),
    )

    # Sampling and loop configuration
    min_per_user = int(cfg["train"].get("record_per_user", 10))
    predict_count = int(cfg["train"].get("prediction_count", 256))
    multi_turn = bool(cfg["train"].get("multi_turn", False))

    metric = str(cfg["train"].get("objective", "auc")).lower()  # "auc" or "acc"
    minimize = False
    best = float("-inf")

    # Checkpoint configuration
    run_id = cfg["train"].get("id", "default")
    ckpt_root = cfg.get("checkpoint", {}).get("root", "./models")
    save_every = int(cfg.get("checkpoint", {}).get("save_every", 25))

    ckpt_dir = os.path.join(ckpt_root, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_ckpt_path = os.path.join(ckpt_dir, f"{cfg['dataset']['name']}_best_model.pt")

    # Validation data (optional)
    val_jsonl = os.path.join(base_dir, "valid.jsonl")
    if not os.path.isfile(val_jsonl):
        val_jsonl = None

    # Main training loop
    epochs = int(cfg["train"].get("epochs", 350))
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

        visible_edges = add_reverse_edges_to_dict(visible_edges, device)

        # Forward pass
        agg_emb = model(graph_data.x_dict, visible_edges)
        user_emb, query_emb, llm_emb = agg_emb["user"], agg_emb["query"], agg_emb["llm"]

        # Loss and training metrics
        loss, acc, auc = compute_pairwise_loss_and_metrics(
            predict_list, predictor, user_emb, query_emb, llm_emb, response_rating, device
        )
        loss.backward()
        optim.step()

        # Validation on JSONL (if available)
        model.eval()
        predictor.eval()
        current_val = None
        if val_jsonl is not None:
            val = evaluate_jsonl(val_jsonl, predictor, agg_emb, user_dict, llm_dict, device, verbose=False)
            current_val = val.get(metric, 0.0)
            best = save_if_best(best_ckpt_path, current_val, best, minimize, epoch, model, predictor, dim, user_dict, metric)

        # Logging and periodic checkpoint (based on epoch index)
        if save_every and (epoch % save_every == 0):
            msg = f"[Epoch {epoch}] loss={loss.item():.4f} acc={acc:.3f} auc={auc:.3f}"
            if current_val is not None:
                msg += f" | val_{metric}={current_val:.3f}"
            print(msg)

            periodic_path = os.path.join(ckpt_dir, f"{cfg['dataset']['name']}_ckpt_{epoch}.pt")
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

    # Load the best model before final evaluation
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        predictor.load_state_dict(ckpt["predictor_state_dict"])
        print(f"[FINAL] Loaded best epoch {ckpt.get('epoch', '?')} {metric.upper()}={ckpt.get('score', 0):.4f}")

    # Final evaluation: prefer test.pt, otherwise fall back to test.jsonl (if present)
    test_jsonl = os.path.join(base_dir, "test.jsonl")
    test_pt = os.path.join(base_dir, "test.pt")

    print("\n[TEST] Running final evaluation...")
    if os.path.isfile(test_pt):
        (t_cfg, t_nodes, t_edges, _t_user, _t_q, _t_r, t_ratings, _t_llm, t_meta, t_device) = build_config(test_pt)
        t_graph = build_hetero_data(t_nodes, t_edges)

        t_graph = T.ToUndirected()(t_graph)
        t_graph = t_graph.to(t_device)

        t_edges_with_rev = add_reverse_edges_to_dict(t_edges, t_device)

        with torch.no_grad():
            t_agg = model(t_graph.x_dict, t_edges_with_rev)

        # Stable iteration order for reproducible metrics
        t_pairs = sorted(list(t_meta))
        _t_loss, t_acc, t_auc = compute_pairwise_loss_and_metrics(
            t_pairs, predictor, t_agg["user"], t_agg["query"], t_agg["llm"], t_ratings, t_device
        )
        print(f"[TEST-pt] acc={t_acc:.4f} | auc={t_auc:.4f}")
    elif os.path.isfile(test_jsonl):
        with torch.no_grad():
            full_train_emb = model(graph_data.x_dict, graph_data.edge_index_dict)

        test_metrics = evaluate_jsonl(test_jsonl, predictor, full_train_emb, user_dict, llm_dict, device, verbose=True)
        print(f"[TEST-jsonl] acc={test_metrics['acc']:.4f} | auc={test_metrics['auc']:.4f}")
    else:
        print("[TEST] No test.jsonl or test.pt found; skipping.")

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
    Command-line entry point for training the HeteroGNN preference model.

    Expects a YAML configuration file passed via ``--config``.
    """
    parser = argparse.ArgumentParser(description="Train HeteroGNN preference model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_with_cfg(cfg)


if __name__ == "__main__":
    main()
