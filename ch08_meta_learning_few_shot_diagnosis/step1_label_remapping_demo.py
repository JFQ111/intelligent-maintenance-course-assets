"""第 8 章 任务一：标签重映射机制演示。

用法：
    python step1_label_remapping_demo.py

本脚本不需要真实数据，也不需要 GPU，专门用于演示和验证元学习中
标签重映射的核心逻辑：

1. 构造一个包含 10 类的模拟数据集（对应 CWRU 10 类子集）。
2. 模拟两轮 episode 采样，展示同一类别在不同 episode 中的标签编号变化。
3. 演示若不做重映射，直接使用原始全局标签会导致什么问题。
4. 对比随机重映射与固定重映射的差异。
"""
from __future__ import annotations

import numpy as np


# ── CWRU 10 类标签名称（与第一章保持一致） ──────────────────────
LABEL_NAMES = [
    "normal",
    "inner_race_007", "ball_007", "outer_race_007",
    "inner_race_014", "ball_014", "outer_race_014",
    "inner_race_021", "ball_021", "outer_race_021",
]
N_CLASSES = len(LABEL_NAMES)  # 10


def simulate_dataset(n_classes: int, n_per_class: int = 50, signal_length: int = 1024,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """生成模拟信号数据集（随机高斯信号，仅用于演示）。"""
    rng = np.random.default_rng(seed)
    signals = rng.standard_normal((n_classes * n_per_class, signal_length)).astype(np.float32)
    labels  = np.repeat(np.arange(n_classes, dtype=np.int64), n_per_class)
    return signals, labels


def sample_episode_without_remapping(
    labels: np.ndarray,
    class_pool: list[int],
    n_way: int,
    k_shot: int,
    n_query: int,
    rng: np.random.Generator,
) -> dict:
    """错误示范：不做标签重映射，直接使用全局标签。"""
    chosen = rng.choice(class_pool, size=n_way, replace=False)
    support_y, query_y = [], []
    for c in chosen:
        support_y.extend([c] * k_shot)
        query_y.extend([c] * n_query)
    return {
        "chosen_classes": chosen.tolist(),
        "support_y_raw": support_y,
        "query_y_raw": query_y,
        "max_label": max(support_y),
        "n_way": n_way,
    }


def sample_episode_with_remapping(
    labels: np.ndarray,
    class_pool: list[int],
    n_way: int,
    k_shot: int,
    n_query: int,
    rng: np.random.Generator,
) -> dict:
    """正确做法：随机选类 → 随机打乱顺序 → 映射为 0..N-1。"""
    chosen = list(rng.choice(class_pool, size=n_way, replace=False))
    rng.shuffle(chosen)  # 打乱顺序，增加任务多样性

    # 建立重映射表
    label_map = {c: i for i, c in enumerate(chosen)}

    support_y, query_y = [], []
    for c in chosen:
        support_y.extend([label_map[c]] * k_shot)
        query_y.extend([label_map[c]] * n_query)

    return {
        "chosen_classes": chosen,
        "label_map": label_map,
        "support_y_remapped": support_y,
        "query_y_remapped": query_y,
        "max_label": n_way - 1,
        "n_way": n_way,
    }


def print_separator(title: str = "") -> None:
    width = 60
    if title:
        print(f"\n{'=' * width}")
        print(f"  {title}")
        print(f"{'=' * width}")
    else:
        print("-" * width)


def main() -> None:
    rng = np.random.default_rng(0)
    signals, labels = simulate_dataset(N_CLASSES, n_per_class=50)
    class_pool = list(range(N_CLASSES))

    N_WAY, K_SHOT, N_QUERY = 5, 5, 15

    # ── 1. 演示不做重映射的问题 ──────────────────────────────────
    print_separator("1. 不做标签重映射的问题演示")
    ep_bad = sample_episode_without_remapping(
        labels, class_pool, N_WAY, K_SHOT, N_QUERY, rng
    )
    print(f"随机选取的 {N_WAY} 个全局类别：{ep_bad['chosen_classes']}")
    print(f"  → 标签名称：{[LABEL_NAMES[c] for c in ep_bad['chosen_classes']]}")
    print(f"支持集标签（直接使用原始标签）：{ep_bad['support_y_raw'][:10]}  ...")
    print(f"支持集标签的最大值：{ep_bad['max_label']}")
    print(f"网络输出层节点数（n_way）：{ep_bad['n_way']}")
    print()
    print("问题：若标签最大值（如 9）>= 输出层节点数（5），")
    print("      CrossEntropyLoss 会报 'Target out of bounds' 错误。")

    # ── 2. 演示正确的标签重映射：第一个 episode ─────────────────
    rng2 = np.random.default_rng(1)
    print_separator("2. 正确的标签重映射：第一个 episode")
    ep1 = sample_episode_with_remapping(
        labels, class_pool, N_WAY, K_SHOT, N_QUERY, rng2
    )
    print(f"随机选取的全局类别（已随机打乱顺序）：{ep1['chosen_classes']}")
    print(f"  → 标签名称：{[LABEL_NAMES[c] for c in ep1['chosen_classes']]}")
    print(f"重映射表（全局标签 → episode 内临时标签）：")
    for global_lbl, local_lbl in ep1['label_map'].items():
        print(f"    {global_lbl:2d} ({LABEL_NAMES[global_lbl]:20s}) → {local_lbl}")
    print(f"支持集标签（重映射后）：{ep1['support_y_remapped'][:10]}  ...")
    print(f"支持集标签的最大值：{ep1['max_label']}（= n_way - 1 = {N_WAY - 1}）")
    print("结论：标签范围 0 到 n_way-1，与输出层节点数完全匹配。")

    # ── 3. 第二个 episode：相同类别会被赋予不同编号 ─────────────
    print_separator("3. 第二个 episode（相同类别 → 不同临时标签）")
    ep2 = sample_episode_with_remapping(
        labels, class_pool, N_WAY, K_SHOT, N_QUERY, rng2
    )
    print(f"随机选取的全局类别：{ep2['chosen_classes']}")
    print(f"重映射表：")
    for global_lbl, local_lbl in ep2['label_map'].items():
        print(f"    {global_lbl:2d} ({LABEL_NAMES[global_lbl]:20s}) → {local_lbl}")

    # 找出两轮都选到的类别，展示编号变化
    common = set(ep1['chosen_classes']) & set(ep2['chosen_classes'])
    if common:
        print(f"\n两轮 episode 共同选到的类别：{sorted(common)}")
        print("  在 episode 1 中的临时标签：", {c: ep1['label_map'][c] for c in common})
        print("  在 episode 2 中的临时标签：", {c: ep2['label_map'][c] for c in common})
        print("结论：同一类别在不同 episode 中编号不同，")
        print("      迫使模型从信号特征本身学习，而非记忆标签编号。")
    else:
        print("两轮 episode 选取了完全不同的类别（也是正常情况）。")

    # ── 4. 小结 ──────────────────────────────────────────────────
    print_separator("4. 标签重映射总结")
    print("① 必要性：episode 随机采样的全局标签不连续，不能直接送入网络。")
    print("② 做法  ：每轮 episode 内部将 n_way 个类别映射为 0..n_way-1。")
    print("③ 范围  ：重映射只在当前 episode 内有效，结束即丢弃。")
    print("④ 效果  ：防止标签记忆，增加任务多样性，强制特征学习。")


if __name__ == "__main__":
    main()
