import os
import shutil
import argparse


def find(images_dir, labels_dir, output_dir):
    if not os.path.isdir(images_dir):
        raise SystemExit(f"[错误] 图片目录不存在: {images_dir}")
    if not os.path.isdir(labels_dir):
        # 不静默：没有标签目录 = 所有图都会被判成"未标注"，通常意味着数据集缺失
        print(f"[警告] 标签目录不存在: {labels_dir} —— 会把该目录下所有图片都当成未标注")

    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for img in sorted(os.listdir(images_dir)):
        if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        label = os.path.splitext(img)[0] + '.txt'
        if not os.path.exists(os.path.join(labels_dir, label)):
            shutil.copy(os.path.join(images_dir, img),
                        os.path.join(output_dir, img))
            count += 1
    print(f"未标注图片已复制 {count} 张到 {output_dir}")
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='找出未标注图片并复制到输出目录')
    parser.add_argument('--images-dir', default='data/images', help='原始图片目录')
    parser.add_argument('--labels-dir', default='data/labels', help='已有标签目录')
    parser.add_argument('--output-dir', default='data/unlabeled', help='未标注图片输出目录')
    args = parser.parse_args()
    find(args.images_dir, args.labels_dir, args.output_dir)
