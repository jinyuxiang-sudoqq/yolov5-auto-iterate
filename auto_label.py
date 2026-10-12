import os
import sys
import shutil
import argparse
import cv2
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 复用 dual_check 里同一套推理前处理 / 标签写入，避免两处实现各改各的漂移
from dual_check import infer, write_label


def auto_label(weights, img_dir, out_dir, conf=0.25, iou_thres=0.45, imgsz=640):
    """单模型预标注：给 img_dir 每张图出一份草稿标签，图片 + 标签成对写到 out_dir。

    ⚠️ 产物是【草稿】，只用来加速人工复核。
       绝不能把 out_dir 直接当训练集 / 直接并入训练集 —— 单模型没有质量闸门。
    """
    if not os.path.isdir(img_dir):
        raise SystemExit(f"[错误] 图片目录不存在: {img_dir}")
    if not os.path.exists(weights):
        raise SystemExit(f"[错误] 权重不存在: {weights}")

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = None
    try:
        from models.common import DetectMultiBackend
        model = DetectMultiBackend(weights, device=device)
        model.eval()
    except Exception as e:
        raise SystemExit(f"[错误] 加载权重失败: {type(e).__name__}: {e}")

    count = 0
    total_boxes = 0
    empty = 0
    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(img_dir, img_name)
        img0 = cv2.imread(img_path)
        if img0 is None:
            print(f"[跳过] 读不出图片: {img_name}")
            continue

        dets = infer(model, img0, device, conf, iou_thres, imgsz)
        stem = os.path.splitext(img_name)[0]
        write_label(os.path.join(out_dir, stem + '.txt'), dets, img0.shape)
        shutil.copy(img_path, os.path.join(out_dir, img_name))
        total_boxes += len(dets)
        if not dets:
            empty += 1
        count += 1

    print(f"预标注完成: {count} 张，共 {total_boxes} 个草稿框（其中 {empty} 张模型没检出任何目标）")
    print(f"草稿目录 → {out_dir}")
    print("下一步：在草稿上人工修正，改完后把「图 + 同名 .txt」成对挪进 approved 目录；")
    print("        不同意/改不好的直接删掉。草稿目录本身不会被并入训练集。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='单模型预标注（产出草稿，需人工修正后才可采用）')
    parser.add_argument('--weights', required=True, help='模型权重路径')
    parser.add_argument('--img-dir', required=True, help='待标注图片目录')
    parser.add_argument('--out-dir', required=True, help='草稿输出目录（图 + 同名标签成对）')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU 阈值')
    parser.add_argument('--imgsz', type=int, default=640, help='推理尺寸')
    args = parser.parse_args()
    auto_label(args.weights, args.img_dir, args.out_dir, args.conf, args.iou, args.imgsz)
