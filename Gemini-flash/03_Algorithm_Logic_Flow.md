### 文件名: 03_Algorithm_Logic_Flow.md

本文件还原 `TensorCrossInterpolation.jl` 项目中 **TCI2**（两点更新交叉插值）算法的核心逻辑流。TCI2 是目前张量压缩领域最稳健的算法之一。

#### 1. 入口函数 (Entrance)

用户通常从以下函数启动计算：
*   **函数名:** `crossinterpolate2`
*   **所在文件:** `src/tensorci2.jl` (第 943 行)
*   **主要参数:** 目标函数 `f`, 每个维度的尺寸 `localdims`, 初始枢轴 `initialpivots`, 以及精度容差 `tolerance`。

#### 2. 执行流程图 (Execution Flow)

基于 `src/tensorci2.jl` 中的 `optimize!` 函数 (第 700 行) 及相关子函数，算法逻辑如下：

1.  **初始化 (Initialization):**
    *   根据 `initialpivots` 构建初始的指标集 `Iset` 和 `Jset`。
    *   计算初始点的函数值以设置 `maxsamplevalue`（用于误差归一化）。

2.  **主迭代循环 (Main Loop):**
    *   **步骤 2.1: 两点扫描 (`sweep2site!`)** (`src/tensorci2.jl` 第 855 行)
        *   算法沿张量链进行前向或后向扫描。
        *   对于每一对相邻站点（Bond），调用 `updatepivots!`。
        *   **局部采样:** 构建局部指标集的笛卡尔积，形成局部矩阵 $\Pi$。
        *   **局部交叉插值:** 对 $\Pi$ 执行 `MatrixLUCI` (LU 分解交叉插值)，更新该位置的枢轴索引。
    *   **步骤 2.2: 全局搜索 (`searchglobalpivots`)** (`src/tensorci2.jl` 第 958 行)
        *   在整个多维空间中寻找当前插值结果与原函数 `f` 误差最大的点。
        *   如果发现误差在 `abstol` 以上的点，将其作为“全局枢轴”加入 `Iset/Jset`。这一步保证了算法不会陷入局部极小值。
    *   **步骤 2.3: 判定终止 (`convergencecriterion`)** (`src/tensorci2.jl` 第 609 行)
        *   检查最近几次迭代的误差是否均小于 `tolerance`。
        *   检查秩是否已达到 `maxbonddim`。

3.  **收尾与清理 (Finalization):**
    *   **单点扫描 (`sweep1site!`)** (`src/tensorci2.jl` 第 402 行)
        *   进行一次单点扫描，精简掉由全局搜索引入的冗余枢轴。
        *   最终生成并存储核心张量 (`sitetensors`)。

#### 3. 数学内核操作 (Mathematical Operations)

在 C++ 移植中，需要实现以下核心数学操作：

| 数学操作 | 实现位置 (证据) | 说明 |
| :--- | :--- | :--- |
| **秩显示 LU 分解** | `src/matrixlu.jl` (`rrlu`) | TCI 的核心逻辑在于寻找准极大子矩阵，通常通过带全主元或部分主元的 LU 分解实现。 |
| **数值稳定求逆** | `src/matrixci.jl` (`AtimesBinv`) | 解决 $A \cdot B^{-1}$ 问题。代码中使用了 `QR` 分解：`QR = qr([A; B]); result = Q[1:m, :] / Q[m+1:end, :]`。 |
| **张量折叠/展开** | `src/tensorci2.jl` (多处 `reshape`) | 将多维张量暂时视为 2D 矩阵以应用线性代数算子。 |
| **收缩运算 (Contraction)** | `src/contraction.jl` | 执行类似于 $C_{ijk} = \sum_{l} A_{ijl} B_{lk}$ 的运算。 |

#### 4. 关键证据引用

*   **执行循环控制:** `src/tensorci2.jl` L768-L831。
*   **枢轴更新逻辑:** `src/tensorci2.jl` L511 (`updatepivots!`)。
*   **误差评估逻辑:** `src/tensorci2.jl` L157 (`pivoterror`)。
