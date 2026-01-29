# TensorCrossInterpolation.jl 终极技术指南 (Ultimate Technical Guide)

**版本**: 1.0
**生成日期**: 2026-01-28
**目标读者**: 核心开发者、算法研究员、高级用户

---

## 目录 (Table of Contents)

1.  [引言 (Introduction)](#1-引言-introduction)
2.  [数学理论基础 (Mathematical Foundations)](#2-数学理论基础-mathematical-foundations)
    *   2.1 张量列 (Tensor Train) 分解
    *   2.2 交叉插值 (Cross Interpolation) 原理
    *   2.3 骨架分解 (Skeleton Decomposition) 与 MaxVol 原理
3.  [项目架构深度解析 (Architecture Deep Dive)](#3-项目架构深度解析-architecture-deep-dive)
    *   3.1 核心文件概览
    *   3.2 类型系统 (Type System)
4.  [核心算法详解 (Core Algorithm Details)](#4-核心算法详解-core-algorithm-details)
    *   4.1 TensorCI2 构造过程
    *   4.2 扫描策略 (Sweeping Strategy)
    *   4.3 局部矩阵分解 (Local Matrix Factorization)
    *   4.4 全局主元搜索 (Global Pivot Search)
    *   4.5 收敛判据 (Convergence Criteria)
5.  [源码级实现分析 (Source Code Analysis)](#5-源码级实现分析-source-code-analysis)
    *   5.1 关键结构体定义
    *   5.2 关键函数逻辑
6.  [扩展与开发指南 (Extension Guide)](#6-扩展与开发指南-extension-guide)
7.  [API 参考 (API Reference)](#7-api-参考-api-reference)

---

## 1. 引言 (Introduction)

`TensorCrossInterpolation.jl` 是一个高性能的 Julia 库，旨在解决高维函数的“维数灾难”问题。通过实现张量交叉插值（Tensor Cross Interpolation, TCI）算法，它能够以指数级减少的存储空间和计算量，来逼近定义在高维网格上的函数或大规模张量。

### 核心价值
-   **Black-Box Interpolation**: 仅需提供函数 $f(x)$ 的求值接口，无需知道函数的解析形式。
-   **Adaptive Rank**: 自动根据精度要求调整张量的秩（Bond Dimension），实现精度与性能的平衡。
-   **High Performance**: 基于 Julia 的高性能实现，利用 BLAS/LAPACK 加速底层矩阵运算。

---

## 2. 数学理论基础 (Mathematical Foundations)

理解代码前，必须掌握其背后的数学原理。

### 2.1 张量列 (Tensor Train) 分解

对于一个 $d$ 维张量 $\mathcal{T} \in \mathbb{R}^{n_1 \times n_2 \times \dots \times n_d}$，其 Tensor Train 分解形式为：
$$ \mathcal{T}(i_1, i_2, \dots, i_d) \approx \sum_{\alpha_0, \dots, \alpha_d} G_1(\alpha_0, i_1, \alpha_1) G_2(\alpha_1, i_2, \alpha_2) \dots G_d(\alpha_{d-1}, i_d, \alpha_d) $$
其中：
-   $G_k$ 是第 $k$ 个核心张量（Core Tensor），大小为 $r_{k-1} \times n_k \times r_k$。
-   $r_k$ 称为第 $k$ 个键维（Bond Dimension）或 TT-Rank。边界条件通常设为 $r_0 = r_d = 1$。
-   这种分解将存储复杂度从 $O(n^d)$ 降低到 $O(d n r^2)$，其中 $r = \max(r_k)$。

### 2.2 交叉插值 (Cross Interpolation) 原理

交叉插值（Cross Interpolation，在矩阵情形下称为 Skeleton Decomposition）的核心思想是：一个低秩矩阵可以被其少量的行和列即其“骨架”（Skeleton）唯一确定。

对于矩阵 $A \in \mathbb{R}^{m \times n}$，若其秩为 $r$，则存在 $r$ 行索引 $I$ 和 $r$ 列索引 $J$，使得：
$$ A \approx C \cdot \hat{A}^{-1} \cdot R $$
其中：
-   $C = A[:, J]$ 是选定的列组成的矩阵。
-   $R = A[I, :]$ 是选定的行组成的矩阵。
-   $\hat{A} = A[I, J]$ 是行列交叉处的 $r \times r$ 子矩阵。

TCI 算法将这一思想推广到 Tensor Train 格式。每个核心张量 $G_k$ 由前一个键的“索引集” $I_{k-1}$ 和后一个键的“索引集” $J_k$ 决定。算法的目标就是动态地寻找最优的 $I_k$ 和 $J_k$ 集合。

### 2.3 MaxVol 原理

如何选择最好的 $I$ 和 $J$？理论告诉我们，应该选择使得子矩阵 $\hat{A}$ 的行列式（或体积 Volume）模最大的那些行列。这就是 **MaxVol (Maximum Volume)** 原理。
在该项目中，为了计算效率，并没有严格求解 MaxVol 问题（这是一个 NP-hard 问题），而是使用了 **Rank-Revealing LU (RRLU)** 分解作为贪心近似。RRLU 通过每次选取剩余矩阵中绝对值最大的元素作为主元，自然地逼近了 MaxVol 的解。

---

## 3. 项目架构深度解析 (Architecture Deep Dive)

### 3.1 核心文件概览

*   **`src/TensorCrossInterpolation.jl`**: 模块入口。
*   **`src/abstracttensortrain.jl`**: 定义 `AbstractTensorTrain` 接口。
*   **`src/tensortrain.jl`**: `TensorTrain` 具体实现（静态数据结构）。
*   **`src/tensorci2.jl`**: `TensorCI2` 具体实现（动态构造算法）。
*   **`src/matrixlu.jl`**: `rrLU` 和 `rrlu!` 实现（底层数学核心）。
*   **`src/matrixluci.jl`**: 将 `rrLU` 包装为适应 TCI 接口的 `MatrixLUCI`。
*   **`src/globalpivotfinder.jl`**: 全局主元搜索策略。

### 3.2 类型系统 (Type System)

```julia
AbstractTensorTrain{V}
├── TensorTrain{V, N}  (存储最终结果，核心是 Array 数组)
└── TensorCI2{V}       (存储中间状态，核心是 Iset, Jset 和 Site Tensors)
```

**`TensorCI2` 的关键字段**:
-   `Iset::Vector{Vector{MultiIndex}}`: 每个 Bond 左侧的基索引集合。每一个 `MultiIndex` 是一个整数向量 `[i_1, i_2, ..., i_k]`。
-   `Jset::Vector{Vector{MultiIndex}}`: 每个 Bond 右侧的基索引集合。每一个 `MultiIndex` 是 `[i_{k+1}, ..., i_d]`。
-   `sitetensors`: 缓存当前的核张量，避免每次都重新计算。
-   `pivoterrors`: 记录每个 Bond 的截断误差，用于判定收敛。

---

## 4. 核心算法详解 (Core Algorithm Details)

### 4.1 TensorCI2 构造过程 (`crossinterpolate2`)

1.  **初始化**:
    -   创建 `TensorCI2` 对象。
    -   使用用户提供的 `initialpivots`（默认为全 1）初始化 `Iset` 和 `Jset`。
    -   调用 `optimize!` 开始迭代。

### 4.2 扫描策略 (Sweeping Strategy)

`optimize!` 函数采用类似 DMRG (Density Matrix Renormalization Group) 的扫描方式。默认策略是 `:backandforth`（前向扫描 $\to$ 后向扫描）。

核心函数 `sweep2site!` (双点扫描)：
它不是一次只更新一个核，而是同时更新两个相邻的核 $G_k, G_{k+1}$。
1.  **构建环境 (Environment)**: 利用左边的 $I_k$ 和右边的 $J_{k+1}$。
2.  **形成局部矩阵 (Local Matrix)**:
    考虑键 $k$ 处的“超矩阵” $\mathcal{M}$。其行索引由 $(I_k, i_k)$ 组成，列索引由 $(i_{k+1}, J_{k+1})$ 组成。
    实际并不生成这个超大矩阵，而是定义一个能够计算 $\mathcal{M}_{row, col}$ 的函数闭包。
3.  **分解 (Factorization)**: 对 $\mathcal{M}$ 进行 RRLU 分解。
    $$ \mathcal{M} \approx L \cdot U $$
    分解产生的新主元（Pivots）决定了键 $k$ 处新的 $I_{k+1}$（即 $J_k$）。
    如果 RRLU 发现秩增加了（即找到了更多显著不为 0 的主元），则 Bond Dimension $r_k$ 会自动增加。

### 4.3 局部矩阵分解 (Local Matrix Factorization)

这是 TCI 的“显微镜”。由 `src/matrixlu.jl` 处理。
-   **输入**: 一个隐式定义的矩阵（函数句柄）。
-   **Full Search**: 计算整个局部矩阵（大小为 $R_{in} d \times d R_{out}$），然后在内存中做 RRLU。对于 $d$ 较小的情况很有效。
-   **Rook Pivoting**: 为了处理大规模局部矩阵，不生成全量矩阵，而是使用 Rook Search（车行搜索）。从一个非零元出发，交替在行和列中找最大值，直到收敛到局部最大值。这在 `arrlu` 函数中实现。

### 4.4 全局主元搜索 (Global Pivot Search)

为了防止算法陷入局部最优（即所有的局部矩阵更新都无法发现某些远处的“尖峰”），需要全局视野。
-   由 `src/globalpivotfinder.jl` 负责。
-   **机制**: 生成随机样本点 $x_{rand}$，比较 $f(x_{rand})$ 和当前近似 $TT(x_{rand})$。
-   **Refinement**: 如果 $\text{error} > \text{tol}$，则以该随机点为起点，进行坐标下降（Coordinate Descent）或简单的局部搜索，找到该区域误差最大的点 $x_{max}$。
-   **Update**: 将 $x_{max}$ 解析为对应的 Left/Right Indices，强制加入到 `Iset` 和 `Jset` 中。

### 4.5 收敛判据 (Convergence Criteria)

算法在以下条件满足时停止：
1.  **误差达标**: 所有 Bond 的 Pivoting Error 和全局采样的误差都小于 `tolerance`。
2.  **秩稳定**: 连续几次迭代秩不再增加。
3.  **无新主元**: 全局搜索找不到新的显著误差主元。

---

## 5. 源码级实现分析 (Source Code Analysis)

### 5.1 `TensorCI2` 结构体
```julia
mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}
    Iset::Vector{Vector{MultiIndex}}  # Left indices bases
    Jset::Vector{Vector{MultiIndex}}  # Right indices bases
    localdims::Vector{Int}            # Physical dimensions
    sitetensors::Vector{Array{ValueType,3}} # Cached cores
    pivoterrors::Vector{Float64}      # Error tracking
    # ...
end
```
**设计意图**: `Iset` 和 `Jset` 是 TCI 的“源数据”，`sitetensors` 是衍生物。只要有 `Iset/Jset` 和函数 $f$，就可以重建 `sitetensors`。

### 5.2 `updatepivots!` (`src/tensorci2.jl`)
这是连接 TCI 逻辑和 Matrix LU 逻辑的桥梁。
```julia
function updatepivots!(tci, b, f, leftorthogonal; ...)
    # 1. 准备扩展的索引集 (User extra sets + Current sets)
    Icombined = ... 
    Jcombined = ...
    
    # 2. 调用 MatrixLUCI 进行分解
    luci = MatrixLUCI(...) 
    
    # 3. 根据 LU 结果更新索引集
    tci.Iset[b+1] = Icombined[rowindices(luci)]
    tci.Jset[b] = Jcombined[colindices(luci)]
    
    # 4. 更新 Tensor Core (left/right factors)
    setsitetensor!(tci, b, left(luci))
    setsitetensor!(tci, b + 1, right(luci))
end
```
这段代码展示了 TCI 如何通过局部更新来迭代地改进 TT 结构。注意 `leftorthogonal` 参数，它决定了我们是在前向扫描（更新 $A$ 为左正交）还是后向扫描。

### 5.3 `rrlu!` (`src/matrixlu.jl`)
核心的数值线性代数实现。
```julia
function rrlu!(A; ...)
    # ...
    while lu.npivot < maxrank
        # 寻找下一个主元 (Greedy selection)
        k = lu.npivot + 1
        newpivot = submatrixargmax(abs2, A, k)
        
        # 将主元交换到对角线 (k, k)
        addpivot!(lu, A, newpivot)
        
        # Schur Complement Update (in-place)
        # A[k+1:end, k+1:end] -= A[k+1:end, k] * A[k, k+1:end]
    end
end
```
实现采用了原地更新（in-place）的方式来减少内存分配，这对于高性能计算至关重要。`submatrixargmax` 被高度优化以利用 CPU 缓存。

---

## 6. 扩展与开发指南 (Extension Guide)

### 6.1 如何添加新的全局主元搜索策略？

如果你想通过贝叶斯优化或遗传算法来寻找主元：
1.  **定义新类型**:
    ```julia
    struct GeneticGlobalPivotFinder <: AbstractGlobalPivotFinder
        # 参数: 种群大小, 变异率等
    end
    ```
2.  **实现调用接口**:
    在 `src/globalpivotfinder.jl` 中（或新文件）实现：
    ```julia
    function (finder::GeneticGlobalPivotFinder)(
        input::GlobalPivotSearchInput{ValueType},
        f,
        abstol::Float64;
        kwargs...
    )::Vector{MultiIndex}
        # 实现你的遗传算法逻辑
        # 返回找到的高误差索引列表
    end
    ```
3.  **使用**: 在调用 `crossinterpolate2` 或 `optimize!` 时传入 `globalpivotfinder=GeneticGlobalPivotFinder(...)`。

### 6.2 如何支持其他类型的分解（如 QR 或 SVD）？

目前的实现强绑定于 LU 分解。若要支持其他分解（虽然 TCI 通常特指基于行列交叉的插值）：
1.  定义新的 `AbstractMatrixCI` 子类型，例如 `MatrixSVDCI`。
2.  实现 `rowindices`, `colindices`, `left`, `right` 等接口。
3.  修改 `_factorize` 函数以分发到新的类型。
    *注意*：传统的 SVD 不直接给出“行列索引”，因此不适合标准的 Cross Interpolation，但在压缩阶段（`compress!`）可以使用。

---

## 7. API 参考 (API Reference)

### 7.1 `crossinterpolate2`
```julia
crossinterpolate2(ValueType, f, localdims, initialpivots; kwargs...)
```
-   **用途**: 构建函数 `f` 的 TCI 近似。
-   **参数**:
    -   `ValueType`: 函数返回值的类型（如 `Float64`）。
    -   `f`: 目标函数，接受 `Vector{Int}` 返回 `ValueType`。
    -   `localdims`: 每一维的大小。
    -   `initialpivots`: 初始猜测点。
-   **关键 Kwargs**:
    -   `tolerance`: 目标精度（绝对误差）。
    -   `maxbonddim`: 最大允许秩。
    -   `maxiter`: 最大扫描次数。

### 7.2 `optimize!`
```julia
optimize!(tci, f; ...)
```
-   **用途**: 对已有的 `tci` 对象进行进一步的扫描优化。
-   **场景**: 当你需要手动控制流程，或者分阶段优化时使用。

### 7.3 `TensorTrain`
```julia
tt(indexset)
```
-   **用途**: 对 TT 进行求值。
-   **复杂度**: $O(d r^2)$。

### 7.4 `rank`
```julia
rank(tt)
```
-   **用途**: 返回 TT 的最大键维。这是衡量压缩效果的关键指标。

---

## 结语

`TensorCrossInterpolation.jl` 展示了现代算法工程的典范：将深奥的数学原理（张量分析）转化为可用的、高性能的代码工具。通过阅读本文档，开发者应具备了深入修改代码、定制算法以及解决复杂应用问题的能力。
