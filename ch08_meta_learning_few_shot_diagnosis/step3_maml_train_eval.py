"""第 8 章 任务三：MAML（一阶近似 FOMAML）元训练与元测试。

用法：
    python step3_maml_train_eval.py --npz-path ./outputs/cwru_12k_drive_end_10class_windows_w1024_s512.npz
    python step3_maml_train_eval.py --demo          # 使用模拟数据

MAML 双层优化结构（教材第八章）：
    内循环（inner loop）：在每个 episode 的支持集上对 deepcopy 的参数
        执行若干步 SGD，得到任务专属参数。
    外循环（outer loop）：用任务专属参数在查询集上计算损失，累加后
        对元初始参数 θ 求梯度并更新（FOMAML：不追溯内循环梯度图）。

注意：完整二阶 MAML 需要 `higher` 库（pip install higher）。
      本脚本默认使用 FOMAML（deepcopy + 标准反向传播），
      在大多数诊断任务中性能与二阶版本相当。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from embed_net import EmbedNetWithHead
from episode_sampler import EpisodeSampler, FewShotDataset


# ── 内循环（快速适应） ─────────────────────────────────────────

def inner_loop(
    model: EmbedNetWithHead,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    alpha: float,
    n_steps: int,
) -> EmbedNetWithHead:
    """在支持集上执行若干步 SGD，返回适应后的参数副本。

    Parameters
    ----------
    model : 当前元初始参数的模型（不修改原始参数）。
    support_x : Tensor, shape (N*K, 1, L)
    support_y : Tensor, shape (N*K,)，已完成标签重映射（0..N-1）。
    alpha : 内循环学习率。
    n_steps : 内循环梯度下降步数。

    Returns
    -------
    fast_model : 适应后的模型副本。
    """
    fast_model = deepcopy(model)          # 不修改元初始参数
    inner_opt  = optim.SGD(fast_model.parameters(), lr=alpha)
    criterion  = nn.CrossEntropyLoss()

    fast_model.train()
    for _ in range(n_steps):
        inner_opt.zero_grad()
        logits = fast_model(support_x)
        loss   = criterion(logits, support_y)
        loss.backward()
        inner_opt.step()

    return fast_model


# ── 外循环（元训练） ───────────────────────────────────────────

def train_maml(
    model: EmbedNetWithHead,
    sampler: EpisodeSampler,
    n_epochs: int,
    episodes_per_epoch: int,
    alpha: float,
    beta: float,
    n_inner_steps: int,
    device: torch.device,
) -> list[dict]:
    """FOMAML 外循环元训练。

    每个 epoch 采样 episodes_per_epoch 个 episode，累加查询集损失后
    对元初始参数做一次外循环更新。
    """
    model = model.to(device)
    meta_opt  = optim.Adam(model.parameters(), lr=beta)
    criterion = nn.CrossEntropyLoss()
    history   = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        meta_opt.zero_grad()
        epoch_meta_loss, epoch_acc = 0.0, 0.0

        for _ in range(episodes_per_epoch):
            ep = sampler.sample()
            sx = torch.from_numpy(ep.support_x).to(device)
            sy = torch.from_numpy(ep.support_y).to(device)
            qx = torch.from_numpy(ep.query_x).to(device)
            qy = torch.from_numpy(ep.query_y).to(device)

            # 内循环：在支持集上快速适应
            fast_model = inner_loop(model, sx, sy, alpha, n_inner_steps)

            # 在查询集上计算元损失（FOMAML：fast_model 的梯度图不追溯内循环）
            fast_model.train()
            logits    = fast_model(qx)
            task_loss = criterion(logits, qy)
            task_loss.backward()  # 梯度累积到 meta_opt 管理的 model.parameters()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                epoch_acc += (preds == qy).float().mean().item()
            epoch_meta_loss += task_loss.item()

        # 外循环参数更新（每 epoch 一次）
        meta_opt.step()

        avg_loss = epoch_meta_loss / episodes_per_epoch
        avg_acc  = epoch_acc       / episodes_per_epoch
        history.append({"epoch": epoch, "loss": avg_loss, "acc": avg_acc})

        if epoch % 10 == 0 or epoch == 1:
            print(f"[MAML]    Epoch {epoch:4d}/{n_epochs} | "
                  f"Meta Loss={avg_loss:.4f} | Query Acc={avg_acc*100:.1f}%")

    return history


# ── 元测试（快速适应 + 评估） ─────────────────────────────────

def eval_maml(
    model: EmbedNetWithHead,
    sampler: EpisodeSampler,
    n_episodes: int,
    alpha: float,
    n_inner_steps: int,
    device: torch.device,
) -> float:
    """在测试 episode 上快速适应后评估准确率。"""
    model.to(device)
    accs = []
    criterion = nn.CrossEntropyLoss()

    for _ in range(n_episodes):
        ep = sampler.sample()
        sx = torch.from_numpy(ep.support_x).to(device)
        sy = torch.from_numpy(ep.support_y).to(device)
        qx = torch.from_numpy(ep.query_x).to(device)
        qy = torch.from_numpy(ep.query_y).to(device)

        # 用测试支持集快速适应
        fast_model = inner_loop(model, sx, sy, alpha, n_inner_steps)

        fast_model.eval()
        with torch.no_grad():
            logits = fast_model(qx)
            preds  = logits.argmax(dim=1)
            acc    = (preds == qy).float().mean().item()
        accs.append(acc)

    return float(np.mean(accs))


# ── 命令行入口 ──────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MAML（FOMAML）元训练与元测试。")
    p.add_argument("--npz-path", default=None)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--n-way",    type=int,   default=5)
    p.add_argument("--k-shot",   type=int,   default=5)
    p.add_argument("--n-query",  type=int,   default=15)
    p.add_argument("--n-epochs", type=int,   default=50)
    p.add_argument("--episodes-per-epoch", type=int, default=20)
    p.add_argument("--n-eval-episodes",    type=int, default=100)
    p.add_argument("--alpha",    type=float, default=0.01,
                   help="内循环学习率（默认 0.01）。")
    p.add_argument("--beta",     type=float, default=1e-3,
                   help="外循环学习率（默认 0.001）。")
    p.add_argument("--n-inner-steps", type=int, default=5,
                   help="内循环梯度下降步数（默认 5）。")
    p.add_argument("--seed",     type=int,   default=42)
    return p


def make_demo_dataset(seed: int = 0) -> FewShotDataset:
    rng = np.random.default_rng(seed)
    n_classes, n_per_class, sig_len = 10, 200, 1024
    signals = rng.standard_normal((n_classes * n_per_class, sig_len)).astype(np.float32)
    labels  = np.repeat(np.arange(n_classes, dtype=np.int64), n_per_class)
    return FewShotDataset(signals, labels)


def main() -> int:
    args   = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.demo or args.npz_path is None:
        print("使用模拟数据（--demo 模式）")
        dataset = make_demo_dataset(args.seed)
    else:
        print(f"加载数据：{args.npz_path}")
        dataset = FewShotDataset.from_npz(Path(args.npz_path))

    print(f"数据集：{dataset.n_classes()} 类，"
          f"每类样本数：{list(dataset.n_samples_per_class().values())[:3]}...")

    train_sampler = EpisodeSampler(
        dataset, n_way=args.n_way, k_shot=args.k_shot,
        n_query=args.n_query, seed=args.seed,
    )
    eval_sampler = EpisodeSampler(
        dataset, n_way=args.n_way, k_shot=args.k_shot,
        n_query=args.n_query, seed=args.seed + 1,
    )

    # MAML 需要带分类头的网络（n_way 个输出节点）
    model = EmbedNetWithHead(n_way=args.n_way, in_channels=1, hidden=64, out_dim=128)

    print(f"\n开始 MAML 元训练：{args.n_way}-way {args.k_shot}-shot，"
          f"alpha={args.alpha}，beta={args.beta}，inner_steps={args.n_inner_steps}")
    history = train_maml(
        model, train_sampler,
        n_epochs=args.n_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        alpha=args.alpha,
        beta=args.beta,
        n_inner_steps=args.n_inner_steps,
        device=device,
    )

    test_acc = eval_maml(
        model, eval_sampler, args.n_eval_episodes,
        alpha=args.alpha, n_inner_steps=args.n_inner_steps,
        device=device,
    )
    print(f"\n[MAML] 元测试准确率（{args.n_eval_episodes} episodes）：{test_acc*100:.2f}%")
    print("注意：--demo 模式使用随机信号，准确率接近随机水平（~20% for 5-way）。")
    print("      使用真实 CWRU 数据（--npz-path）可获得有意义的诊断准确率。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
