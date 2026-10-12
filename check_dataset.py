import os
import sys
import argparse


def load_yaml_names(data_yaml):
    try:
        import yaml as _yaml
    except ImportError:
        raise SystemExit("[中止] 缺少 pyyaml，无法读取数据集 yaml（pip install pyyaml）")
    with open(data_yaml, 'r', encoding='utf-8') as f:
        d = _yaml.safe_load(f) or {}
    names = d.get('names')
    if names is None:
        raise SystemExit(f"[中止] {data_yaml} 里没有 names 字段")
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in dict(names).items()}


def load_weight_names(weights):
    import torch
    # 只加载可信的、自己训出来的权重：weights_only=False 会执行 pickle，
    # 对来路不明的 .pt 文件不要这么干。
    try:
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(weights, map_location='cpu')
    model = ckpt.get('model') if isinstance(ckpt, dict) else None
    names = getattr(model, 'names', None)
    if names is None:
        return None
    return {int(k): str(v) for k, v in dict(names).items()}


def check(data_yaml, weights):
    """校验数据集 yaml 的类别定义与权重一致 —— 防止微调时标签索引错位。"""
    if not os.path.exists(data_yaml):
        raise SystemExit(f"[中止] 数据集 yaml 不存在: {data_yaml}")
    if not os.path.exists(weights):
        raise SystemExit(f"[中止] 权重不存在: {weights}")

    y = load_yaml_names(data_yaml)
    w = load_weight_names(weights)
    if w is None:
        print("[警告] 权重未暴露 names，跳过类别一致性校验")
        return

    if y != w:
        print(f"  yaml ({data_yaml}):")
        print(f"    {y}")
        print(f"  权重 ({weights}):")
        print(f"    {w}")
        raise SystemExit("[中止] 数据集 yaml 的 names 与权重不一致 —— 训练时标签索引会错位！")

    print(f"OK  数据集 yaml 与权重类别一致（{len(w)} 类），索引不会错位")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='校验数据集 yaml 与权重的类别定义是否一致')
    parser.add_argument('--data', required=True, help='数据集 yaml 路径')
    parser.add_argument('--weights', required=True, help='权重路径')
    args = parser.parse_args()
    check(args.data, args.weights)
