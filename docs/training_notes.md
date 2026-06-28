# Training and Experiment Notes

This document records commands and file-management conventions used during development.

## Recommended logging

Use `tee` so that terminal output is preserved:

```bash
mkdir -p work_dir/P2BFoV

CUDA_VISIBLE_DEVICES=0,1 \
bash tools/dist_train.sh \
    configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py \
    2 \
    --work-dir work_dir/P2BFoV/ \
    2>&1 | tee work_dir/P2BFoV/train.log
```

For inference or validation:

```bash
python <inference_entry>.py \
    <arguments> \
    2>&1 | tee work_dir/P2BFoV/inference.log
```

## Files that must be archived after each experiment

- Effective config file.
- Training log.
- Validation or inference log.
- Checkpoint files.
- Prediction JSON.
- Metric output.
- Representative visualizations.
- A short text file describing the code change.

## Temporary modules

Modules under `temp_change/` are experimental implementations waiting to be tested. Copy only the required module into the active code path and record the corresponding commit hash before training.

## Process inspection

View processes owned by the current user:

```bash
ps -u "$USER" -f
```

Filter training processes:

```bash
ps -u "$USER" -f | grep -E "python.*train.py"
```

Filter inference processes:

```bash
ps -u "$USER" -f | grep -E "python.*test.py"
```

## Process termination

Terminate only matching training processes:

```bash
pkill -u "$USER" -f "python.*train.py"
```

Terminate only matching inference processes:

```bash
pkill -u "$USER" -f "python.*test.py"
```

Use `kill -9 <PID>` only when a process cannot be terminated normally. Avoid killing all processes owned by the user on a shared server unless you have verified that this is safe.

## Suggested experiment record

```text
Experiment:
Date:
Git commit:
Config:
Checkpoint:
GPU:
Core modification:
Training status:
mIoU:
mAP:
AP50:
Observed issue:
Next action:
```
