import os
import sys
import shutil
import argparse
import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def infer(model, img0, device, conf=0.25, iou_thres=0.45, imgsz=640):
    img = letterbox(img0, imgsz, stride=model.stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = non_max_suppression(model(img), conf, iou_thres)
    results = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf_v, cls in det:
                results.append([float(v) for v in xyxy] + [int(cls), float(conf_v)])
    return results


def match(det_a, det_b, iou_thres=0.5):
    """严格一一对应：类别相同 + IoU >= 阈值，且两边数量一致才算"一致"。"""
    used = [False] * len(det_b)
    for a in det_a:
        best_iou, best_j = 0, -1
        for j, b in enumerate(det_b):
            if used[j] or a[4] != b[4]:
                continue
            v = iou(a[:4], b[:4])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou < iou_thres:
            return False
        used[best_j] = True
    for j in range(len(det_b)):
        if not used[j]:
            return False
    return True


def write_label(out_path, dets, img_shape):
    h, w = img_shape[:2]
    lines = []
    for d in dets:
        x1, y1, x2, y2, cls, _ = d
        # 夹到图内，避免归一化坐标越界
        x1, x2 = max(0.0, min(x1, float(w))), max(0.0, min(x2, float(w)))
        y1, y2 = max(0.0, min(y1, float(h))), max(0.0, min(y2, float(h)))
        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))


def check_names(model_a, model_b, path_a, path_b):
    """两个模型必须落在同一类别集合上，否则直接拦死。

    match() 是按【类别索引】比较的：如果 A 的索引 0 是 red_light、B 的索引 0 是 dog，
    两边"索引 0 一致"会被判为通过，进而写出完全错误的标签 —— 而且是静默的。
    """
    na = getattr(model_a, 'names', None)
    nb = getattr(model_b, 'names', None)
    if na is None or nb is None:
        print("[警告] 权重未暴露 .names，跳过类别集合一致性校验（按索引比较仍有风险）")
        return
    na, nb = dict(na), dict(nb)
    if na != nb:
        raise SystemExit(
            "[中止] 两个模型的类别集合不一致 —— 按索引交叉验证会产出错误标签！\n"
            f"  A ({path_a}):\n    {na}\n"
            f"  B ({path_b}):\n    {nb}\n"
            "通常是因为两个 exp 用了不同的数据集/类别定义；请只与同一血统的 exp 交叉验证。"
        )
    print(f"类别集合一致性校验通过（{len(na)} 类）")


def dual_check(weights_a, weights_b, img_dir, agree_dir, review_dir):
    os.makedirs(agree_dir, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)

    if not os.path.isdir(img_dir):
        raise SystemExit(f"[错误] 待标注图片目录不存在: {img_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    model_a = DetectMultiBackend(weights_a, device=device)
    model_b = DetectMultiBackend(weights_b, device=device)
    model_a.eval()
    model_b.eval()

    check_names(model_a, model_b, weights_a, weights_b)

    agree = 0
    review = 0
    empty = 0
    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(img_dir, img_name)
        img0 = cv2.imread(img_path)
        if img0 is None:
            continue

        det_a = infer(model_a, img0, device)
        det_b = infer(model_b, img0, device)

        stem = os.path.splitext(img_name)[0]
        if match(det_a, det_b):
            # 一致：图片 + 同名标签成对放进 agree_dir（YOLOv5 训练必须成对）
            write_label(os.path.join(agree_dir, stem + '.txt'), det_a, img0.shape)
            shutil.copy(img_path, os.path.join(agree_dir, img_name))
            agree += 1
            if not det_a:
                empty += 1
        else:
            # 不一致：图片放进 review_dir，人工复核通过后补同名 .txt
            shutil.copy(img_path, os.path.join(review_dir, img_name))
            review += 1
            print(f"[需复核] {img_name}: A={len(det_a)} B={len(det_b)}")

    total = agree + review
    print(f"\n总计 {total} 张 | 一致 {agree}（其中双方均无目标 {empty}）| 待复核 {review}")
    if total > 0:
        print(f"人工复核比例: {review / total * 100:.1f}%")
    print(f"一致数据（图+标签）→ {agree_dir}")
    print(f"待复核图片 → {review_dir}（同意就补同名 .txt，不同意就删图）")


def parse_opt():
    parser = argparse.ArgumentParser(description='双模型一致性校验')
    parser.add_argument('--model-a', required=True, help='较新的模型权重路径')
    parser.add_argument('--model-b', required=True, help='较旧的模型权重路径')
    parser.add_argument('--img-dir', required=True, help='待标注图片目录')
    parser.add_argument('--agree-dir', required=True, help='一致数据输出目录（图片+标签）')
    parser.add_argument('--review-dir', required=True, help='待复核图片输出目录')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    dual_check(
        weights_a=opt.model_a,
        weights_b=opt.model_b,
        img_dir=opt.img_dir,
        agree_dir=opt.agree_dir,
        review_dir=opt.review_dir
    )
