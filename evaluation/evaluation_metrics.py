import os
import json
import numpy as np
import nibabel as nib
from sklearn.metrics import jaccard_score
from medpy.metric.binary import dc

def compute_metrics_per_slice(pred_path, gt_path):
    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)

    pred = pred_nii.get_fdata().astype(np.uint8)
    gt = gt_nii.get_fdata().astype(np.uint8)

    dice_per_slice = []
    iou_per_slice = []

    for i in range(pred.shape[2]):
        pred_slice = pred[:, :, i].flatten()
        gt_slice = gt[:, :, i].flatten()

        if np.sum(pred_slice) == 0 and np.sum(gt_slice) == 0:
            dice_score = 1.0
            iou_score = 1.0
        else:
            try:
                dice_score = dc(pred_slice, gt_slice)
            except:
                dice_score = 0.0
            try:
                iou_score = jaccard_score(gt_slice, pred_slice, average='binary')
            except:
                iou_score = 0.0

        dice_per_slice.append(dice_score)
        iou_per_slice.append(iou_score)

    return dice_per_slice, iou_per_slice

# 路径配置
pred_dir = '/data1/zhangfeiyan/U-Mamba/data/inference_result/Dataset020_nozaitong/case1'
gt_dir = '/data1/zhangfeiyan/U-Mamba/data/nnUNet_raw/Dataset020_nozaitong/imagesTS-label'

results = {}

for case_id in range(120, 134):
    case_name = f"nozaitong_{case_id:04d}"
    pred_path = os.path.join(pred_dir, f"{case_name}.nii.gz")
    gt_path = os.path.join(gt_dir, f"{case_name}.nii.gz")

    if os.path.exists(pred_path) and os.path.exists(gt_path):
        dice, iou = compute_metrics_per_slice(pred_path, gt_path)
        results[case_name] = {
            "dice_per_slice": dice,
            "iou_per_slice": iou
        }
    else:
        results[case_name] = {
            "error": "Missing prediction or ground truth file"
        }

# 保存结果到 JSON 文件
output_json = "/data1/zhangfeiyan/U-Mamba/data/inference_result/Dataset020_nozaitong/case1/slice_metrics.json"
with open(output_json, 'w') as f:
    json.dump(results, f, indent=4)

output_json
