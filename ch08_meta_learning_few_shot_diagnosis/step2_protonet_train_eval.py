"""第 8 章 任务二：原型网络（ProtoNet）元训练与元测试。

用法：
    python step2_protonet_train_eval.py --npz-path ./outputs/cwru_12k_drive_end_10class_windows_w1024_s512.npz
    python step2_protonet_train_eval.py --demo          # 使用模拟数据，无需真实 npz

ProtoNet 核心机制：
    1. 嵌入函数 f_θ 把输入信号映射到特征空间。
    2. 支持集中每类样本的嵌入均值即为该类原型向量。
    3. 查询样本与各原型的欧氏距离决定分类概率（负距离 → softmax）。
    4. 推理阶段无需梯度更新，只需重新计算原型。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from embed_net import EmbedNet
from episode_sampler import EpisodeSampler, FewShotDataset


# ── ProtoNet ────────────────────────────────────────────────────

def euclidean_distance(query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """计算查询集与原型之间的欧氏距离矩阵。

    Parameters
    ----------
    query : Tensor, shape (n_query, d)
    prototypes : Tensor, shape (n_way, d)

    Returns
    -------
    dists : Tensor, shape (n_query, n_way)
    """
    # (n_query, 1, d) - (1, n_way, d) = (n_query, n_way, d)
    diff = query.unsqueeze(1) - prototypes.unsqueeze(0)
    return (diff ** 2).sum(dim=-1)  # (n_query, n_way)


def compute_prototypes(support_embeddings: torch.Tensor,
                       support_y: torch.Tensor,
                       n_way: int) -> torch.Tensor:
    """对每个类别的支持集嵌入取均值，得到原型向量。

    Parameters
    ----------
    support_embeddings : Tensor, shape (N*K, d)
    support_y : Tensor, shape (N*K,)
    n_way : int

    Returns
    -------
    prototypes : Tensor, shape (n_way, d)
    """
    d = support_embeddings.shape[-1]
    prototypes = torch.zeros(n_way, d, device=support_embeddings.device)
    for c in range(n_way):
        mask = (support_y == c)
        prototypes[c] = support_embeddings[mask].mean(dim=0)
    return prototypes


def protonet_episode_loss(
    encoder: EmbedNet,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    n_way: int,
) -> tuple[torch.Tensor, float]:
    """执行一次 ProtoNet episode 的前向过程，返回损失和准确率。"""
    # 嵌入
    support_emb = encoder(support_x)          # (N*K, d)
    query_emb   = encoder(query_x)            # (N*Q, d)

    # 原型
    prototypes = compute_prototypes(support_emb, support_y, n_way)  # (N, d)

    # 负距离作为 logits → softmax → 交叉熵
    dists  = euclidean_distance(query_emb, prototypes)  # (N*Q, N)
    logits = -dists
    loss   = nn.CrossEntropyLoss()(logits, query_y)

    # 准确率
    preds   = logits.argmax(dim=1)
    acc     = (preds == query_y).float().mean().item()
    return loss, acc


# ── 训练与评估 ──────────────────────────────────────────────────

def train_protonet(
    encoder: EmbedNet,
    sampler: EpisodeSampler,
    n_epochs: int,
    episodes_per_epoch: int,
    lr: float,
    device: torch.device,
) -> list[dict]:
    encoder = encoder.to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    history = []

    for epoch in range(1, n_epochs + 1):
        encoder.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        for _ in range(episodes_per_epoch):
            ep = sampler.sample()
            sx = torch.from_numpy(ep.support_x).to(device)
            sy = torch.from_numpy(ep.support_y).to(device)
            qx = torch.from_numpy(ep.query_x).to(device)
            qy = torch.from_numpy(ep.query_y).to(device)

            optimizer.zero_grad()
            loss, acc = protonet_episode_loss(
                encoder, sx, sy, qx, qy, sampler.n_way
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc  += acc

        avg_loss = epoch_loss / episodes_per_epoch
        avg_acc  = epoch_acc  / episodes_per_epoch
        history.append({"epoch": epoch, "loss": avg_loss, "acc": avg_acc})

        if epoch % 10 == 0 or epoch == 1:
            print(f"[ProtoNet] Epoch {epoch:4d}/{n_epochs} | "
                  f"Loss={avg_loss:.4f} | Acc={avg_acc*100:.1f}%")

    return history


def eval_protonet(
    encoder: EmbedNet,
    sampler: EpisodeSampler,
    n_episodes: int,
    device: torch.device,
) -> float:
    encoder.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = sampler.sample()
            sx = torch.from_numpy(ep.support_x).to(device)
            sy = torch.from_numpy(ep.support_y).to(device)
            qx = torch.from_numpy(ep.query_x).to(device)
            qy = torch.from_numpy(ep.query_y).to(device)
            _, acc = protonet_episode_loss(
                encoder, sx, sy, qx, qy, sampler.n_way
            )
            accs.append(acc)
    return float(np.mean(accs))


# ── 命令行入口 ──────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ProtoNet 元训练与元测试。")
    p.add_argument("--npz-path", default=None,
                   help="第一章生成的窗口级 npz 文件路径。")
    p.add_argument("--demo", action="store_true",
                   help="使用模拟数据运行演示，无需真实 npz。")
    p.add_argument("--n-way",    type=int,   default=5)
    p.add_argument("--k-shot",   type=int,   default=5)
    p.add_argument("--n-query",  type=int,   default=15)
    p.add_argument("--n-epochs", type=int,   default=50)
    p.add_argument("--episodes-per-epoch", type=int, default=20)
    p.add_argument("--n-eval-episodes",    type=int, default=100)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--seed",     type=int,   default=42)
    return p


def make_demo_dataset(seed: int = 0) -> FewShotDataset:
    """生成用于演示的模拟数据集（10 类，每类 200 个样本）。"""
    rng = np.random.default_rng(seed)
    n_classes, n_per_class, sig_len = 10, 200, 1024
    signals = rng.standard_normal((n_classes * n_per_class, sig_len)).astype(np.float32)
    labels  = np.repeat(np.arange(n_classes, dtype=np.int64), n_per_class)
    return FewShotDataset(signals, labels)


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据
    if args.demo or args.npz_path is None:
        print("使用模拟数据（--demo 模式）")
        dataset = make_demo_dataset(args.seed)
    else:
        print(f"加载数据：{args.npz_path}")
        dataset = FewShotDataset.from_npz(Path(args.npz_path))

    print(f"数据集：{dataset.n_classes()} 类，"
          f"每类样本数：{list(dataset.n_samples_per_class().values())[:3]}...")

    # 构造采样器
    train_sampler = EpisodeSampler(
        dataset, n_way=args.n_way, k_shot=args.k_shot,
        n_query=args.n_query, seed=args.seed,
    )
    eval_sampler = EpisodeSampler(
        dataset, n_way=args.n_way, k_shot=args.k_shot,
        n_query=args.n_query, seed=args.seed + 1,
    )

    # 训练
    encoder = EmbedNet(in_channels=1, hidden=64, out_dim=128)
    print(f"\n开始 ProtoNet 元训练：{args.n_way}-way {args.k_shot}-shot")
    history = train_protonet(
        encoder, train_sampler,
        n_epochs=args.n_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        lr=args.lr,
        device=device,
    )

    # 评估
    test_acc = eval_protonet(encoder, eval_sampler, args.n_eval_episodes, device)
    print(f"\n[ProtoNet] 元测试准确率（{args.n_eval_episodes} episodes）：{test_acc*100:.2f}%")
    print("注意：--demo 模式使用随机信号，准确率接近随机水平（~20% for 5-way）。")
    print("      使用真实 CWRU 数据（--npz-path）可获得有意义的诊断准确率。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
