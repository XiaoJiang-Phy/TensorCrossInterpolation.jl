### 文件名: 01_Project_Structure.md

本文件旨在梳理 `TensorCrossInterpolation.jl` 项目的物理文件结构、各模块的功能职责以及项目的外部依赖，为 C++ 移植工作提供宏观视图。

#### 1. 文件清单 (File List)

遍历 `src/` 目录，该项目由以下 Julia 源文件组成：

| 文件名 | 核心职责概述 |
| :--- | :--- |
| `TensorCrossInterpolation.jl` | 项目主入口，定义 `module`，管理依赖导入、导出接口及子文件包含 (`include`)。 |
| `abstractmatrixci.jl` | 定义矩阵交叉插值 (Matrix CI) 的抽象类型及通用接口（如 `getindex`, `localerror`）。 |
| `matrixci.jl` | 实现核心 `MatrixCI` 结构体，提供矩阵级别的插值逻辑、枢轴（Pivot）添加及评估。 |
| `matrixaca.jl` | 实现自适应交叉近似 (Adaptive Cross Approximation, ACA) 算法，用于矩阵的低秩逼近。 |
| `matrixlu.jl` | 提供基于 LU 分解的矩阵分解工具（如 `rrlu` - 秩显示 LU 分解）。 |
| `matrixluci.jl` | 结合 LU 分解实现的交叉插值算法。 |
| `abstracttensortrain.jl` | 定义张量列 (Tensor Train, TT) 的抽象基类，包含 `evaluate`, `norm`, `sum` 等通用操作。 |
| `tensortrain.jl` | 实现具体的 `TensorTrain` (即 MPS, Matrix Product State) 存储结构及其基础代数运算。 |
| `tensorci1.jl` | 实现 **TCI1** 算法：基于单点更新（1-site update）的张量交叉插值。 |
| `tensorci2.jl` | 实现 **TCI2** 算法：基于两点更新（2-site update）的张量交叉插值，通常更健壮。 |
| `globalpivotfinder.jl` | 提供用于初始化算法的初始枢轴搜索策略。 |
| `globalsearch.jl` | 在全空间搜索误差最大点的逻辑，用于辅助枢轴更新。 |
| `indexset.jl` | 管理多维索引（Multi-Index）与线性索引之间的双向映射。 |
| `sweepstrategies.jl` | 定义 TCI 迭代过程中的扫描策略（前向、后向、往复）。 |
| `batcheval.jl` | 提供张量列的高效批量评估（Batch Evaluation）逻辑。 |
| `cachedfunction.jl` | 函数包装器，缓存昂贵的黑盒函数调用结果。 |
| `cachedtensortrain.jl` | 对 TT 对象执行缓存优化。 |
| `contraction.jl` | 实现核心的张量收缩运算（等价于高级矩阵乘法）。 |
| `integration.jl` | 基于 TCI 的数值积分实现（例如利用 QuadGK）。 |
| `conversion.jl` | 不同张量表示格式之间的转换。 |
| `util.jl` | 通用辅助函数（如错误计算、索引操作）。 |

#### 2. 角色映射 (Role Mapping)

根据代码内容，项目的功能模块可划分为四个层次：

*   **基础支撑层 (Infrastructure):**
    *   `indexset.jl`, `util.jl`: 提供索引管理和通用算法辅助。
    *   `sweepstrategies.jl`: 定义迭代控制逻辑。
*   **线性代数层 (Linear Algebra Core):**
    *   `matrixci.jl`, `matrixaca.jl`, `matrixlu.jl`: 这些文件处理 2D 矩阵的低秩分解，是 TCI 算法在低维空间的退化形式或计算组件。C++ 移植时需重点参考 `AtimesBinv` 等数值稳定操作。
*   **张量数据结构层 (Tensor Data Structures):**
    *   `abstracttensortrain.jl`, `tensortrain.jl`: 定义了张量列（MPS）的内存布局。在 C++ 中，这对应于一个 `std::vector<xtensor>` 或 `std::vector<Eigen::Tensor>` 结构的容器。
*   **算法逻辑层 (TCI Algorithms):**
    *   `tensorci1.jl`, `tensorci2.jl`: 整个项目的灵魂。包含交叉插值的核心迭代循环。
    *   `globalpivotfinder.jl`, `globalsearch.jl`: 负责在迭代中寻找改进插值精度的“点”。

#### 3. 依赖分析 (Dependency Analysis)

该项目的 `using`/`import` 语句揭示了其移植到 C++ 时必需的功能库：

| Julia 依赖 | 对应的 C++ 选型建议 | 用途 |
| :--- | :--- | :--- |
| `LinearAlgebra` | **Eigen** / **Armadillo** / **LAPACK** | 提供 QR 分解、SVD 分解、LU 分解、矩阵逆及基础行列操作。 |
| `Random` | `std::random` | 随机初始化枢轴。 |
| `EllipsisNotation` | 需手动实现索引切片逻辑 (如 `xtensor` 的 `view`) | 处理多维数组的变长切片（如 `A[:, i, :]`）。 |
| `QuadGK` | **Boost.Math** / **GSL** | 在 `integration.jl` 中用于数值积分。若不实现积分功能可忽略。 |
| `BitIntegers` | `std::uint64_t` 或 `boost::multiprecision` | 处理大索引或特定格式的位操作。 |

**C++ 移植核心建议：**
针对 C++ 工程师，最关键的第三方库选择是 **Eigen**。由于 Julia 代码中大量使用了 `QR` 分解来实现数值稳定的 `A * inv(B)`，C++ 侧应准备好对应的 `HouseholderQR` 或 `ColPivHouseholderQR` 工具。此外，张量操作（3维及以上数组）推荐使用 **xtensor**，它能很好地模拟 Julia 的广播和切片语法。
