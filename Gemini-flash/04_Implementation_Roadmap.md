### 文件名: 04_Implementation_Roadmap.md

本文件为 C++ 工程师制定的 `TensorCrossInterpolation.jl` 移植路线图，包含模块拆分建议及潜在的技术难点预警。

#### 1. 模块拆解方案 (C++ Module Decomposition)

建议将整个项目拆解为以下 6 个独立的 C++ 模块，按依赖顺序实现：

| 模块名称 | 对应 Julia 文件 | 关键类/函数 | 职责 |
| :--- | :--- | :--- | :--- |
| **L1: 数学支撑库** | `matrixlu.jl`, `matrixci.jl` | `rrlu`, `AtimesBinv` | 基于 Eigen 封装秩显示 LU 和数值稳定的矩阵求逆。 |
| **L2: 索引管理** | `indexset.jl`, `util.jl` | `IndexSet`, `MultiIndex` | 实现多维坐标 `std::vector<int>` 到线性索引的映射。 |
| **L3: 张量容器** | `tensortrain.jl` | `TensorTrain` | 支持 3 阶张量数组的存储、IO 以及基础的 `evaluate` 乘法链。 |
| **L4: 矩阵近似内核** | `matrixaca.jl`, `matrixluci.jl`| `MatrixACA`, `MatrixLUCI` | 实现 2D 矩阵的自适应交叉近似，这是 TCI 的局部计算引擎。 |
| **L5: TCI 核心算法** | `tensorci2.jl` | `sweep2site!`, `updatepivots!` | 实现前向/后向扫描逻辑，管理指标集的增量更新。 |
| **L6: 全局优化器** | `globalpivotfinder.jl` | `searchglobalpivots` | 实现基于启发式（如 Floating Zone）的误差最值点搜索。 |

#### 2. 技术难点与预警 (Technical Challenges)

在翻译 Julia 代码至 C++ 时，需特别注意以下“坑点”：

*   **1. 动态类型与分发 (Dynamic Dispatch):**
    *   **现象:** Julia 频繁使用 `Union{Vector{Int}, NTuple{N, Int}}`。
    *   **对策:** 在 C++ 中，请统一使用 `std::vector<int>` 或 `Eigen::ArrayXi`。避免过度使用模板多态，除非对性能有极致要求。
*   **2. 指标集增量更新的代价:**
    *   **现象:** Julia 的 `pushunique!` 和 `setdiff` 非常方便。
    *   **对策:** 在 C++ 中频繁操作 `std::vector` 的插入和删除会导致大量内存移动。建议使用 `std::set` 或 `std::unordered_set` 维护指标唯一性，仅在数值计算前转换为连续内存。
*   **3. 广播与切片 (Broadcasting & Slicing):**
    *   **现象:** 代码中充满 `A[:, i, :]` 这种切片操作。
    *   **对策:** Eigen 原生对多维张量切片支持有限。强烈建议引入 **xtensor**，它提供了接近 Julia/NumPy 的切片语法和惰性计算广播功能。
*   **4. 闭包与回调函数 (Closures):**
    *   **现象:** 算法核心接受一个函数 `f`。
    *   **对策:** C++ 侧应使用 `std::function<double(const std::vector<int>&)>` 或 lambda 表达式。注意线程安全，因为 `optimize!` 过程中可能会涉及并行的函数评估（Batch Evaluation）。
*   **5. 数值稳定性:**
    *   **核心警告:** 交叉插值对病态矩阵非常敏感。务必严格转换 `AtimesBinv` 中的 QR 分解逻辑，不要直接调用 `Matrix::inverse()`，否则在秩较高时误差会迅速发散。

#### 3. 移植第一阶段目标 (Milestone 1)

建议首先实现一个简化的 **MatrixCI**（2D 版本）。如果您的系统能成功对一个已知低秩矩阵（如 $M_{ij} = \sin(i) + \cos(j)$）进行 1e-12 精度以上的插值还原，则证明基础数学库（L1-L4）已正确通过校验。
