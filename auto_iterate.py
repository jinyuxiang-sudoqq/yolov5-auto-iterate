import os
import re
import shutil
import argparse
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../yolov5
PARENT = os.path.dirname(BASE)                                       # exp* 默认所在的上级目录

IMG_EXTS = ('.jpg', '.jpeg', '.png')

# ---- 流水线工作目录（都在 yolov5/data/ 下）--------------------------------
#   auto_agreed   : 双模型严格一致（图片+标签）   → 会被并入训练集
#   need_review   : 双模型不一致（只有图片）      → 供预标注 / 人工复核
#   review_drafts : 单模型草稿（图片+草稿标签）   → 只加速人工，【绝不】并入训练集
#   approved      : 人工改完并确认采用（图片+标签）→ 会被并入训练集
AGREE_DIR = os.path.join('data', 'auto_agreed')
REVIEW_DIR = os.path.join('data', 'need_review')
DRAFT_DIR = os.path.join('data', 'review_drafts')
APPROVED_DIR = os.path.join('data', 'approved')


def list_exps(exp_root):
    """扫描 exp 根目录，找出所有 exp 文件夹，按数字排序。

    exp_root 可指定（多任务隔离）：不同数据集的 exp 各自放在自己的目录下，
    避免不同类别集合的任务被混在一起交叉验证。
    """
    exps = []
    if not os.path.isdir(exp_root):
        return exps
    for name in os.listdir(exp_root):
        if not name.startswith('exp'):
            continue
        weight = os.path.join(exp_root, name, 'weights', 'best.pt')
        if os.path.exists(weight):
            # 提取数字，exp -> 1, exp2 -> 2
            m = re.match(r'exp(\d*)$', name)
            if m:
                num = int(m.group(1)) if m.group(1) else 1
                exps.append((num, name, weight))
    exps.sort(key=lambda x: x[0])
    return exps


def run(cmd, allow_fail=False):
    """执行命令。默认失败即中止，避免在坏状态上继续往下跑。"""
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        msg = f"命令失败 (exit {r.returncode}): {cmd}"
        if allow_fail:
            print(f"[警告] {msg}")
            return False
        raise SystemExit(f"[中止] {msg}")
    return True


def merge_labeled(src_dirs, dst_images, dst_labels):
    """把「图片 + 同名 .txt」成对的数据并入训练集 —— 迭代闭环的关键一步。

    某张图旁边有同名 .txt 才算通过（dual_check 自动写的，或人工复核后自己补/改的）；
    没有 .txt 的图片一律跳过。返回并入张数。

    注意：src_dirs 只应传「已经过质量闸门」的目录（auto_agreed / approved）。
          单模型草稿目录绝不能传进来。
    """
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)
    added = 0
    for src in src_dirs:
        if not os.path.isdir(src):
            continue
        for name in sorted(os.listdir(src)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMG_EXTS:
                continue
            lab = os.path.join(src, stem + '.txt')
            if not os.path.exists(lab):
                continue
            shutil.copy(os.path.join(src, name), os.path.join(dst_images, name))
            shutil.copy(lab, os.path.join(dst_labels, stem + '.txt'))
            added += 1
    return added


def iterate(rounds, data_yaml, img_size=640, batch=16, epochs=50,
            merge_images=os.path.join('data', 'train', 'images'),
            merge_labels=os.path.join('data', 'train', 'labels'),
            exp_root=None):
    exp_root = os.path.abspath(exp_root) if exp_root else PARENT

    # 先把工作目录建出来，让整个流程可见
    for d in (AGREE_DIR, REVIEW_DIR, DRAFT_DIR, APPROVED_DIR):
        os.makedirs(d, exist_ok=True)

    def weight_of(name):
        return os.path.join(exp_root, name, 'weights', 'best.pt')

    print(f"exp 根目录 : {exp_root}")
    print(f"数据集 yaml: {data_yaml}")
    print(f"并入目标   : {merge_images} / {merge_labels}")

    for i in range(rounds):
        exps = list_exps(exp_root)
        if len(exps) < 2:
            print(f"需要至少两个 exp 才能做交叉验证，当前只有 {len(exps)} 个")
            return

        old_exp = exps[-2][1]   # 次新
        new_exp = exps[-1][1]   # 最新
        print(f"\n===== 第 {i+1} 轮迭代 =====")
        print(f"交叉验证: {new_exp} vs {old_exp}")

        # 1. 找未标注图
        run('python auto_iterate/find_unlabeled.py')

        # 2. 双模型交叉验证
        #    dual_check 内部会先校验两个权重的类别集合是否一致，不一致直接中止
        run(
            f'python auto_iterate/dual_check.py '
            f'--model-a "{weight_of(new_exp)}" '
            f'--model-b "{weight_of(old_exp)}" '
            f'--img-dir data/unlabeled '
            f'--agree-dir {AGREE_DIR} '
            f'--review-dir {REVIEW_DIR}'
        )

        # 2.5 单模型预标注：给待复核的图出草稿框
        #     只是加速人工（从"画框"变"改框"），成功与否都不影响主流程
        run(
            f'python auto_iterate/auto_label.py '
            f'--weights "{weight_of(new_exp)}" '
            f'--img-dir {REVIEW_DIR} '
            f'--out-dir {DRAFT_DIR}',
            allow_fail=True,
        )

        # 3. 等人工复核
        print("\n---- 人工复核 ----")
        print(f"  1) 打开 {DRAFT_DIR}（图已带草稿框），逐张修正")
        print(f"  2) 改完的把「图 + 同名 .txt」成对挪进 {APPROVED_DIR}")
        print("  3) 不同意 / 改不好的直接删掉")
        print(f"  注意：{DRAFT_DIR} 本身不会被并入训练集，只有挪进 {APPROVED_DIR} 的才算数")
        input("完成后按回车继续...")

        # 3.5 把通过的新数据并入训练集
        #     只读 auto_agreed（双模型一致）+ approved（人工确认）；
        #     刻意【不读】review_drafts —— 草稿未经确认就进训练集 = 静默污染
        added = merge_labeled([AGREE_DIR, APPROVED_DIR], merge_images, merge_labels)
        print(f"已并入训练集 {added} 张 → {merge_images} / {merge_labels}")
        if added == 0:
            print("[警告] 本轮没有任何新数据并入，训练会照跑，但等价于原地踏步")

        # 3.8 训练前校验：数据集 yaml 的 names 必须与权重的 names 一致
        #     （否则微调时标签索引错位，训练照跑但结果是错的）
        run(f'python auto_iterate/check_dataset.py '
            f'--data "{data_yaml}" --weights "{weight_of(new_exp)}"')

        # 4. 微调训练
        next_num = exps[-1][0] + 1
        next_name = f'exp{next_num}'
        run(
            f'python train.py '
            f'--img {img_size} --batch {batch} --epochs {epochs} '
            f'--data "{data_yaml}" '
            f'--weights "{weight_of(new_exp)}" '
            f'--name {next_name} --project "{exp_root}"'
        )

        print(f"第 {i+1} 轮完成，产出 {next_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="自动迭代训练")
    parser.add_argument('--rounds', type=int, default=3, help='迭代轮数')
    parser.add_argument('--data', type=str, required=True, help='数据集 yaml 路径')
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--merge-images', type=str,
                        default=os.path.join('data', 'train', 'images'),
                        help='训练集图片目录：通过的新数据并入这里')
    parser.add_argument('--merge-labels', type=str,
                        default=os.path.join('data', 'train', 'labels'),
                        help='训练集标签目录：通过的新标签并入这里')
    parser.add_argument('--exp-root', type=str, default=None,
                        help='exp* 所在目录（默认 yolov5 的上一级）。多任务时各自指定，避免不同类别集合混在一起')
    args = parser.parse_args()

    iterate(args.rounds, args.data, args.img, args.batch, args.epochs,
            args.merge_images, args.merge_labels, args.exp_root)
