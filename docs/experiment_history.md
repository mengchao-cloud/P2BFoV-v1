# P2DNet Development and Experiment History

This document consolidates the early point-to-BFoV development log and the later proposal-head experiment history. The original early notes are preserved verbatim in [`BFoVlog_original.md`](BFoVlog_original.md).

---

## 1. Early development timeline

### 2025-12-22 — Initial positive proposal generation

**Goal**

Complete the initial generation of positive spherical proposals by rewriting:

```python
def gen_proposals_from_cfg(gt_points, proposal_cfg, img_meta):
    ...
```

**Main decisions**

- Treat all proposal tensors as BFoV parameters during the first refactoring stage, so that the overall training pipeline can be traced before replacing every planar operation.
- Defer spherical feature extraction until the proposal-generation data flow is confirmed.
- Revisit the conversion from ground-truth boxes to point annotations after checking the dataset input pipeline.
- Replace planar scale definitions with angular BFoV scales.
- Plan to use:

```python
base = 1.0  # degree
base_scales = [5, 10, 20, 40, 90, 180]
```

**Geometric constraint**

A spherical proposal does not require conventional image-boundary clipping. Its primary validity constraint is that the horizontal and vertical fields of view should not exceed \(180^\circ\).

---

### 2025-12-23 — Spherical center jittering

**Goal**

Resolve longitude-latitude transformations during proposal-center jittering. All spherical computations use radians internally.

**Main observations**

- In the original planar implementation, center jittering is performed independently along horizontal and vertical image axes.
- On the sphere, east-west displacement changes both longitude and latitude.
- Left and right jittered points share the same latitude and are symmetric in longitude relative to the original center.
- North-south displacement changes latitude while preserving longitude, except when pole-crossing handling is required.

These observations later became the basis of the four-direction geodesic jittering module used in the fine refinement stage.

---

### 2025-12-24 — Rewriting coarse proposal generation

**Goal**

Finish the spherical rewrite of `gen_proposals_from_cfg`.

**Implementation notes**

- Remove the planar `cut_mode` setting from the configuration after the spherical path is stable.
- Reconsider calls such as:

```python
base_proposals = bbox_cxcywh_to_xyxy(base_proposals)
```

because BFoV proposals are represented by center coordinates and angular fields of view rather than planar corner coordinates.
- Add a spherical coordinate constraint function:

```python
def constrain_spherical_coords(centers):
    ...
```

- Disable the original symmetric image-boundary clipping logic, because it is not directly applicable to the sphere.

---

### 2025-12-28 — BFoV proposal representation

**Representation decision**

Planar boxes are normally displayed through top-left and bottom-right corners. BFoVs are instead represented by spherical center coordinates and angular field-of-view parameters.

The proposal generator returns:

```text
base_proposal_list: [N, 5, 4]
proposals_valid_list: [N, 5, 1]
```

where:

- `N` is the number of annotated centers in the batch.
- `5` corresponds to the original center and four jittered centers.
- The four proposal parameters describe the spherical center and BFoV size.
- The validity tensor indicates whether each proposal is available for subsequent processing.

The code initially preserved the original framework's tensor interfaces to minimize disruption to the training pipeline.

---

### 2025-12-29 — Negative proposal generation

**Goal**

Rewrite:

```python
def gen_negative_proposals(
    gt_points,
    proposal_cfg,
    aug_generate_proposals,
    img_meta,
):
    ...
```

**Planned changes**

- Replace planar overlap computation with spherical IoU.
- Compute pairwise overlap between positive and negative BFoV candidates.
- Replace randomly generated planar corner boxes with randomly generated spherical centers and BFoV angles.
- Retain the default negative proposal count of approximately 500 while validating the sampling distribution.

---

### 2025-12-30 — Fine proposal generation

**Goal**

Rewrite:

```python
def gen_fine_proposals(gt_points, proposal_cfg, img_meta):
    ...
```

**Main decisions**

- Preserve configurable fine proposal generation during early debugging.
- Support proposal generation based on BFoV angular perturbations.
- Keep the option of later replacing simple Cartesian weighting with spherical weighted aggregation.
- Avoid conventional image-boundary clipping for spherical proposals.

---

### 2026-01-02 — ERP boundary handling and fine BFoV generation

**Completed**

- Added ERP horizontal boundary cycling logic.
- Completed an early version of fine BFoV generation.

**Pending architectural changes**

Circular padding was considered for preprocessing, backbone, neck, and potentially proposal or RoI modules:

```python
conv_cfg = dict(type="Conv", padding_mode="circular")
```

The change was postponed until the detector and head data flows had been validated.

**Representation issue**

The planar conversion:

```python
base_boxes_ = bbox_xyxy_to_cxcywh(base_boxes)
```

required additional review because spherical proposals are not naturally represented as planar corner boxes.

---

### 2026-01-10 — BFoV mask generation

**Goal**

Implement and accelerate:

```python
generate_bfov_masks(...)
```

**Status at this stage**

The main panorama-aware pipeline had been rewritten once, including:

- Positive proposal generation.
- Negative proposal generation.
- Fine proposal generation.
- Spherical IoU.
- Weighted proposal fusion.
- BFoV mask generation.
- Proposal feature extraction.

The next step was to trace the complete dataset input and training pipeline, because several operations depended on the exact dataset metadata and annotation representation.

---

### 2026-01-17 — End-to-end pipeline verification

**Goal**

Run through the complete training and evaluation flow.

The evaluation configuration still required verification:

```python
evaluation = dict(
    interval=12,
    metric="bbox",
    save_result_file=work_dir + "_" + str(test_scale) + "_latest_result.json",
    do_first_eval=False,
    do_final_eval=True,
)
```

The main concern was ensuring that output files and metrics remained correct after replacing planar boxes with BFoV representations.

---

### 2026-01-22 — Training, inference, and evaluation refactoring

The training, inference, and evaluation logic had been rewritten and reviewed once. The next step was to trace the runtime commands and test whether the complete pipeline could run on a local GPU.

---

### 2026-01-31 — Scale and IoU experiments

**Changes**

- Tested feature-scale categories of `0.5`, `1`, and `2`.
- Replaced the previous overlap implementation with:

```text
sphiou_efficient_pop
```

**Later conclusion**

The feature-level selection strategy based only on scale categories was abandoned because its logic did not align well with spherical proposal geometry. The subsequent development direction used spherical loss and, later, tangent-plane-based spherical RoI extraction.

---

## 2. Proposal-head experiment history

### head0

- Initial implementation.
- Accuracy was approximately `0.015`.
- Several implementation issues were suspected and required systematic debugging.

### head1

Changes relative to `head0`:

- Removed the left-right feature merging logic.
- Added instance reuse.
- Significantly reduced the risk of GPU memory overflow.

### head2

Changes relative to `head1`:

- Added spherical weighted aggregation.
- A spherical loss was considered for later versions.

### head2-1

Changes relative to `head1`:

- Added spherical weighted aggregation.
- Retained instance reuse to reduce GPU memory consumption.

### head3

Changes relative to `head2-1`:

- Reduced mask resolution.
- Lowered GPU memory consumption during mask processing.

### head4

Changes relative to `head2-1`:

- Revised the scale configuration.
- Avoided additional padding while retaining cross-boundary analysis.
- Adjusted scales to multiples of 64 so that scaling and feature extraction could be performed without extra padding to multiples of 32.
- Located the configuration controlling the number of PBR-stage proposals.

### head4-1

Changes relative to `head2-1`:

- Removed multi-scale training and used a fixed scale.
- Changed feature filtering to use `pad_H × pad_W`.

### head4-2

Changes relative to `head3`:

- Used a debugging-oriented training setting.
- Disabled flipping.
- Trained for seven epochs.

### head5 — planned rewrite

Changes relative to `head4`:

- Planned modification of `mask_to_bbox`.
- For very large fields of view, preserve the main object region instead of directly enclosing the full mask.
- For large regions, select the largest valid left or right bounding box.
- The corresponding configuration may also require adjustment.

### head6

Changes relative to `head4-2`:

- Changed mask-generation and mask-filtering logic.
- For each mask, estimated `max_thickness`.
- Removed mask regions whose thickness was smaller than `0.3 × max_thickness`.
- Limitation: this strategy cannot correctly handle regions crossing the ERP boundary.

### head7

Major revision:

- Removed circular convolution.
- Removed spherical weighting.
- Identified an error in converting pseudo boxes to the final output.
- Corrected the effect of scale factors that enlarged boxes back to the original image size.
- Adopted the `head6` `mask_to_bbox` logic.
- Partially relaxed the scale configuration and enlarged the range of base scales.
- Cleaned up previously mixed code paths so that modifications could take effect reliably.

### head7-1

Changes relative to `head7`:

- Added mean-IoU output.
- Expanded the range and number of base IoU settings.

### head7-2

Changes relative to `head7-1`:

- Reintroduced circular convolution through configuration.
- The result became slightly worse.

### head8

Changes relative to previous versions:

- Replaced the previous proposal feature extraction strategy with tangent-plane-based spherical feature extraction.
- This version corresponds to the spherical RoI direction used in the current paper framework.

---

## 3. Current technical direction

The current manuscript and code direction are centered on:

- Spherical BFoV proposal generation around point annotations.
- Shared-weight original and scale-rotated branches.
- Tangent-plane-based spherical RoI feature extraction.
- FPN level assignment according to spherical region scale.
- MIL-based proposal scoring.
- Top-\(k\) spherical weighted aggregation.
- Four-direction geodesic spherical jittering.
- Multi-scale fine proposal refinement.
- Negative proposal supervision.
- Scale-Rotation Score Consistency loss.

This section should be updated whenever a new stable implementation replaces an earlier experimental head.
