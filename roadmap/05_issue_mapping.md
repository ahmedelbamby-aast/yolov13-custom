# 05 Issue Mapping

## Objective

Map recurring community issues to concrete engineering actions.

## Mapped Issues

- OBB support confusion and runtime errors (e.g. issue threads similar to #62, #74)
  - Action: add v13-obb configs + examples + OBB metric compatibility fix.

- Segmentation setup errors (e.g. issue threads similar to #62, #72)
  - Action: add v13-seg configs + dataset schema validator + one-command smoke script.

- Pose dataset/keypoint schema errors (e.g. issue threads similar to #75)
  - Action: enforce `kpt_shape` validation, add clear docs for multi-instance labels.

- Environment/precision instability reports
  - Action: publish recommended env matrix, lock tested versions, provide diagnostic script.

## Traceability Matrix

```mermaid
flowchart TB
    I1[OBB Issue Cluster] --> A1[v13-obb configs]
    I1 --> A2[OBB metrics key alignment]
    I2[Segmentation Issue Cluster] --> A3[v13-seg configs]
    I2 --> A4[segment data preflight]
    I3[Pose Issue Cluster] --> A5[v13-pose configs]
    I3 --> A6[keypoint schema preflight]
    I4[Env Instability] --> A7[version matrix + diagnostics]
```
