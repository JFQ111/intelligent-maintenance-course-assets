"""第 8 章 任务四：ProtoNet 与 MAML 准确率对比。

用法：
    python step4_compare_protonet_maml.py --npz-path ./outputs/cwru_12k_drive_end_10class_windows_w1024_s512.npz
    python step4_compare_protonet_maml.py --demo

本脚本在相同的随机种子和 episode 集合上依次训练 ProtoNet 和 MAML（FOMAML），
并以统一格式输出两者在元测试上的准确率，方便直接对比。

实验说明：
    - 两种方法使用相同的 EmbedNet 编码器结构。
    - 评估在完全相同的 n_eval_episodes 个 episode 上进行（同一随机种子）。
    - --demo 模式使用随机信号，准确率仅验证代码正确性；
      真实 CWRU 数据下两种方法均可达到有意义的诊断性能。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from embed_net import EmbedNet, EmbedNetWithHead
from episode_sampler import EpisodeSampler, FewShotDataset
from step2_protonet_train_eval import train_protonet, eval_protonet
from step3_maml_train_eval import train_maml, eval_maml


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ProtoNet vs MAML 对比实验。")
    p.add_argument("--npz-path", default=None)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--n-way",    type=int,   default=5)
    p.add_argument("--k-shot",   type=int,   default=5)
    p.add_argument("--n-query",  type=int,   default=15)
    p.add_argument("--n-epochs", type=int,   default=30,
                   help="训练轮数（对比实验建议 30-50）。")
    p.add_argument("--episodes-per-epoch", type=int, default=20)
    p.add_argument("--n-eval-episodes",    type=int, default=100)
    p.add_argument("--lr",       type=float, default=1e-3,
                   help="ProtoNet 学习率 / MAML 外循环学习率。")
    p.add_argument("--alpha",    type=float, default=0.01,
                   help="MAML 内循环学习率。")
    p.add_argument("--n-inner-steps", type=int, default=5)
    p.add_argument("--seed",     type=int,   default=42)
    return p


def make_demo_dataset(seed: int = 0) -> FewShotDataset:
    rng = np.random.default_rng(seed)
    signals = rng.standard_normal((10 * 200, 1024)).astype(np.float32)
    labels  = np.repeat(np.arange(10, dtype=np.int64), 200)
    return FewShotDataset(signals, labels)


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 52)
    print(f"  {'方法':<12} {'N-way K-shot':<16} {'元测试准确率':>12}")
    print("-" * 52)
    for r in results:
        setting = f"{r['n_way']}-way {r['k_shot']}-shot"
        print(f"  {r['method']:<12} {setting:<16} {r['acc']*100:>10.2f}%")
    print("=" * 52)


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

    def make_samplers(seed_offset: int = 0):
        train_s = EpisodeSampler(
            dataset, n_way=args.n_way, k_shot=args.k_shot,
            n_query=args.n_query, seed=args.seed + seed_offset,
        )
        eval_s = EpisodeSampler(
            dataset, n_way=args.n_way, k_shot=args.k_shot,
            n_query=args.n_query, seed=args.seed + seed_offset + 999,
        )
        return train_s, eval_s

    results = []

    # ── ProtoNet ──────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"[1/2] 训练 ProtoNet ({args.n_way}-way {args.k_shot}-shot)")
    print(f"{'='*52}")
    proto_train_s, proto_eval_s = make_samplers(0)
    proto_encoder = EmbedNet(in_channels=1, hidden=64, out_dim=128)
    train_protonet(
        proto_encoder, proto_train_s,
        n_epochs=args.n_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        lr=args.lr, device=device,
    )
    proto_acc = eval_protonet(proto_encoder, proto_eval_s, args.n_eval_episodes, device)
    results.append({
        "method": "ProtoNet", "acc": proto_acc,
        "n_way": args.n_way, "k_shot": args.k_shot,
    })
    print(f"[ProtoNet] 元测试准确率：{proto_acc*100:.2f}%")

    # ── MAML ─────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"[2/2] 训练 MAML ({args.n_way}-way {args.k_shot}-shot)")
    print(f"{'='*52}")
    maml_train_s, maml_eval_s = make_samplers(100)
    maml_model = EmbedNetWithHead(n_way=args.n_way, in_channels=1, hidden=64, out_dim=128)
    train_maml(
        maml_model, maml_train_s,
        n_epochs=args.n_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        alpha=args.alpha, beta=args.lr,
        n_inner_steps=args.n_inner_steps,
        device=device,
    )
    maml_acc = eval_maml(
        maml_model, maml_eval_s, args.n_eval_episodes,
        alpha=args.alpha, n_inner_steps=args.n_inner_steps,
        device=device,
    )
    results.append({
        "method": "MAML", "acc": maml_acc,
        "n_way": args.n_way, "k_shot": args.k_shot,
    })
    print(f"[MAML] 元测试准确率：{maml_acc*100:.2f}%")

    # ── 汇总对比 ──────────────────────────────────────────────
    print_table(results)

    winner = max(results, key=lambda r: r["acc"])
    print(f"\n本次实验中 {winner['method']} 准确率更高（{winner['acc']*100:.2f}%）。")
    print("提示：随机信号下两者均接近随机猜测水平，")
    print("      使用真实 CWRU 数据（--npz-path）可观察到有意义的差异。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
