# YOLOv13 Architecture and Mathematical Enhancement Plan

## Scope and intent
This document analyzes the current YOLOv13 architecture in this repository and proposes a dense, engineering-oriented upgrade plan (YOLOv13-v2).

Focus:
- architecture changes
- mathematical formulation changes
- expected impact on memory, speed, and quality
- expected KV cache hit/loss behavior in attention-heavy paths

References in this codebase:
- `ultralytics/cfg/models/v13/yolov13_2.yaml`
- `ultralytics/nn/modules/block.py`
- `ultralytics/nn/modules/conv.py`

---

## 1) Current YOLOv13 architecture (as implemented)

### 1.1 Macro graph

```mermaid
flowchart TD
    I[Input] --> S1[Conv P1/2]
    S1 --> S2[Conv P2/4]
    S2 --> S3[DSC3k2]
    S3 --> S4[Conv P3/8]
    S4 --> S5[DSC3k2]
    S5 --> S6[DSConv P4/16]
    S6 --> S7[A2C2f x4]
    S7 --> S8[DSConv P5/32]
    S8 --> S9[A2C2f x4]

    S5 --> H0
    S7 --> H0
    S9 --> H0

    H0[HyperACE] --> U1[Upsample]
    H0 --> D1[DownsampleConv]

    S7 --> T12[FullPAD Tunnel]
    H0 --> T12

    S5 --> T13[FullPAD Tunnel]
    U1 --> T13

    S9 --> T14[FullPAD Tunnel]
    D1 --> T14

    T14 --> U2[Upsample]
    U2 --> C17[Concat T12]
    C17 --> N17[DSC3k2]
    N17 --> T18[FullPAD with H0]

    N17 --> U3[Upsample]
    U3 --> C21[Concat T13]
    C21 --> N21[DSC3k2]
    U1 --> P1x1[Conv1x1]
    N21 --> T23[FullPAD with P1x1]

    T23 --> D2[Conv downsample]
    D2 --> C26[Concat T18]
    C26 --> N26[DSC3k2]
    N26 --> T27[FullPAD with H0]

    N26 --> D3[Conv downsample]
    D3 --> C30[Concat T14]
    C30 --> N30[DSC3k2]
    N30 --> T31[FullPAD with D1]

    T23 --> O3[Detect P3]
    T27 --> O4[Detect P4]
    T31 --> O5[Detect P5]
```

### 1.2 HyperACE internals

```mermaid
flowchart LR
    P3[P3 feat] --> F[FuseModule]
    P4[P4 feat] --> F
    P5[P5 feat] --> F

    F --> CV1[1x1 Conv to 3c]
    CV1 --> CK[Chunk y0 y1 y2]

    CK --> B1[C3AH branch 1]
    CK --> B2[C3AH branch 2]
    CK --> LO[Low-order branch DSC3k stack]

    B1 --> CAT[Concat]
    B2 --> CAT
    CK --> CAT
    LO --> CAT

    CAT --> CV2[1x1 Conv out]
```

### 1.3 Adaptive hypergraph flow

```mermaid
flowchart LR
    X[Tokens BxNxD] --> AGen[AdaHyperedgeGen]
    AGen --> A[A matrix BxNxE]
    A --> V2E[He = A^T X]
    V2E --> EProj[Edge MLP]
    EProj --> E2V[Xnew = A He]
    E2V --> NProj[Node MLP]
    NProj --> Out[Residual + X]
```

Current equations:
- `A = softmax(logits, dim=1)`
- `He = A^T X`
- `Xnew = A He`
- `Y = MLP(Xnew) + X`

### 1.4 AAttn and FullPAD

```mermaid
flowchart TD
    Xin[Input feature] --> QK[QK 1x1]
    Xin --> V[V 1x1]
    V --> PE[Depthwise PE]

    QK --> Split[Split Q/K]
    Split --> Attn[Flash or SDPA]
    V --> Attn

    Attn --> R[Reshape back]
    R --> Add[Add with PE]
    PE --> Add
    Add --> Proj[1x1 proj]

    X0[Original] --> PAD
    X1[Enhanced] --> PAD
    PAD[FullPAD: x0 + gate*x1]
```

---

## 2) Core weaknesses to improve

### 2.1 Scalar FullPAD gate is under-expressive
Current fusion uses one scalar parameter for all channels and all spatial locations.

Cons:
- cannot adaptively suppress noisy channels
- cannot spatially route where enhancement is useful

### 2.2 Fusion alignment is too static
`FuseModule` currently uses avgpool/nearest before concat.

Cons:
- no learned offset/alignment
- misalignment error is pushed downstream

### 2.3 HyperACE dual C3AH branch redundancy risk
Two high-order branches ingest near-identical signals.

Cons:
- branches can collapse to correlated representations
- capacity is spent without guaranteed diversity

### 2.4 Hypergraph propagation lacks explicit normalization
Current propagation does not explicitly normalize by node/edge degree matrices.

Cons:
- output scale varies with sequence length and assignment density
- stability can degrade across scale/augmentation regimes

### 2.5 Assignment normalization axis is hard to interpret
Current `softmax(..., dim=1)` normalizes across nodes.

Cons:
- per-edge competition behavior is less direct
- harder to tune sparsity and load balancing

### 2.6 No anti-collapse regularizers for edge usage
No explicit penalties to avoid dead edges or over-dominant edges.

Cons:
- lower effective hypergraph capacity
- higher run-to-run variability

### 2.7 P3/P4/P5 only detection
No optional P2 path in default design.

Con:
- tiny-object recall ceiling on some datasets

### 2.8 AAttn policy still coarse
Even with shape-safe improvements, precision/kernel policy is mostly binary by backend availability.

Con:
- not fully optimized for stability/locality tradeoffs

---

## 3) Proposed YOLOv13-v2 changes

### 3.1 Adaptive FullPAD gating (channel/spatial)

#### Proposed architecture

```mermaid
flowchart LR
    X0[x0] --> Cat
    X1[x1] --> Cat
    Cat[Concat] --> GAP[GlobalAvgPool]
    GAP --> MLP[MLP]
    MLP --> Gc[Channel gate gc]

    Cat --> ConvS[3x3 Conv]
    ConvS --> Gs[Spatial gate gs]

    X1 --> Mul[x1 * gc * gs]
    Gc --> Mul
    Gs --> Mul

    X0 --> Add
    Mul --> Add
    Add[Output]
```

#### Math effect
- moves fusion from rank-1 scalar modulation to channel/spatial-conditioned modulation
- better signal routing and reduced underfitting in neck fusion

#### Expected impact
- Memory: +1% to +3%
- Throughput: -1% to -3%
- Accuracy robustness: positive
- KV cache hits/losses: near-neutral (not direct QKV path)

### 3.2 Alignment-aware multi-scale fusion

#### Proposed architecture

```mermaid
flowchart TD
    P3 --> A3[Learned alignment to P4 grid]
    P4 --> I4[Identity or light refine]
    P5 --> A5[Learned alignment to P4 grid]
    A3 --> Cat[Concat]
    I4 --> Cat
    A5 --> Cat
    Cat --> Fuse[1x1/3x3 fuse]
```

#### Math effect
- reduces aliasing and phase mismatch before high-order modeling
- improves coherence entering HyperACE

#### Expected impact
- Memory: +3% to +8%
- Throughput: -3% to -8%
- Accuracy: moderate gain
- KV cache hits/losses: small gain from less corrective attention work

### 3.3 Branch diversity regularization in HyperACE

#### Proposed training graph

```mermaid
flowchart TD
    X[Shared input] --> B1[C3AH-1]
    X --> B2[C3AH-2]
    B1 --> Z1[z1]
    B2 --> Z2[z2]
    Z1 --> DivL[Ldiv]
    Z2 --> DivL
    DivL --> Total[Task loss + lambda*Ldiv]
```

#### Math effect
- encourages complementary high-order features
- prevents branch collapse

#### Expected impact
- Memory: ~0 runtime
- Throughput: ~0 to -1% training
- Quality: positive on complex scenes
- KV cache hits/losses: neutral

### 3.4 Degree-normalized hypergraph convolution

#### Proposed equations
- `Dv(i,i)=sum_e A(i,e)`
- `De(e,e)=sum_i A(i,e)`
- `Xnew = Dv^(-1/2) A De^(-1) A^T Dv^(-1/2) X W`

#### Graph view

```mermaid
flowchart LR
    X[X] --> Dv[Compute Dv]
    X --> A[A]
    A --> De[Compute De]
    A --> N1[Dv^-1/2 * A]
    De --> N2[* De^-1]
    N1 --> N2
    N2 --> N3[* A^T * Dv^-1/2]
    N3 --> Y[Normalized propagation]
```

#### Math effect
- better-conditioned propagation operator
- lower scale drift across token counts

#### Expected impact
- Memory: +2% to +5%
- Throughput: -2% to -5%
- Stability: strong gain
- KV cache hits/losses: slight raw overhead possible, but better effective utilization via fewer unstable iterations

### 3.5 Assignment strategy upgrade

#### Candidate strategies

```mermaid
flowchart TD
    L[Logits] --> ESoft[Softmax over edges]
    L --> Sink[Sinkhorn-lite balancing]
    L --> TopK[Top-k sparse + renorm]
    ESoft --> Eval[Evaluate quality/speed/memory/cache]
    Sink --> Eval
    TopK --> Eval
```

#### Math effect
- better assignment interpretability
- sparse routing can improve efficiency and specialization

#### Expected impact
- Memory: -10% to +5% (top-k vs Sinkhorn)
- Throughput: +5% to -6%
- Accuracy: often positive with proper regularization
- KV cache hits/losses: top-k sparse usually increases hit ratio and reduces cache losses

### 3.6 Hyperedge usage regularizers

#### Suggested losses
- entropy term on assignments
- batch-level edge load balancing
- dead-edge penalty

#### Ma


---

## 8) Implementation status (phase 1)

Implemented in codebase:

- Adaptive FullPAD fusion:
  - `FullPAD_Tunnel` upgraded from scalar-only residual to scalar-scaled channel/spatial adaptive gating.
- Hypergraph routing upgrades:
  - `AdaHyperedgeGen` now supports `normalize` in `{node, edge, sinkhorn}`.
  - Optional sparse routing with `topk` masking and renormalization.
- Degree-normalized propagation:
  - `AdaHGConv` now supports degree-normalized incidence and configurable fallback.
- Module plumbing:
  - `AdaHGComputation`, `C3AH`, `HyperACE` signatures extended to pass `normalize`, `topk`, `degree_norm`.
  - `parse_model` logic in `ultralytics/nn/tasks.py` updated accordingly.
- YAML defaults:
  - New v13 _2 YAML variants carry upgraded defaults (`normalize="edge"`, `topk=0`, `degree_norm=True`) while base YAMLs stay unchanged.

Files touched in phase 1:

- `ultralytics/nn/modules/block.py`
- `ultralytics/nn/tasks.py`
- `ultralytics/cfg/models/v13/yolov13_2.yaml`
- `ultralytics/cfg/models/v13/yolov13s_2.yaml`
- `ultralytics/cfg/models/v13/yolov13l_2.yaml`
- `ultralytics/cfg/models/v13/yolov13x_2.yaml`

Not yet implemented (phase 2+):

- alignment-aware FuseModule replacement
- explicit HyperACE branch-diversity loss wiring in training objective
- optional P2 detection-head profile
- AAttn windowed-policy path
