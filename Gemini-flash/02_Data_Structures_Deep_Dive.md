### 文件名: 02_Data_Structures_Deep_Dive.md

本文件深入解析 `TensorCrossInterpolation.jl` 中的核心数据结构，揭示其内存布局与逻辑含义，并为 C++ 工程师提供等效的类设计建议。

#### 1. 核心 Struct 定义

项目中最重要的三个数据结构分别是用于 2D 矩阵的 `MatrixCI`，以及用于多维张量的 `TensorTrain` 和 `TensorCI1/2`。

##### A. `MatrixCI{T}` (矩阵交叉插值状态)
这是 TCI 算法的基石，代表对一个矩形矩阵的低秩分解状态。
*   **rowindices (`Vector{Int}`):** 选定的行枢轴索引集 $\mathcal{I}$。
*   **colindices (`Vector{Int}`):** 选定的列枢轴索引集 $\mathcal{J}$。
*   **pivotcols (`Matrix{T}`):** 枢轴列矩阵，即原矩阵在所有行但在 $\mathcal{J}$ 列上的采样。
*   **pivotrows (`Matrix{T}`):** 枢轴行矩阵，即原矩阵在 $\mathcal{I}$ 行但在所有列上的采样。

##### B. `TensorTrain{ValueType, N}` (张量列/MPS)
这是插值结果的存储格式。
*   **sitetensors (`Vector{Array{ValueType, N}}`):** 一个由三阶张量（Core Tensors）组成的数组。每个张量的维数通常为 `(link_left, physical_dim, link_right)`。

##### C. `TensorCI1{ValueType}` (TCI1 算法状态)
*   **Iset/Jset (`Vector{IndexSet}`):** 存储左侧和右侧的多维索引集合。
*   **T (`Vector{Array{T, 3}}`):** 核心张量数组。
*   **P (`Vector{Matrix{T}}`):** 链接矩阵（Link Matrices），用于连接相邻的核心张量。
*   **Pi (`Vector{Matrix{T}}`):** 局部“全”矩阵的采样，用于寻找新的枢轴。

#### 2. Julia 语法解码

为了在 C++ 中准确还原，需要理解以下 Julia 特性：
1.  **参数化类型 `{T}`:** 类似于 C++ 的 `template <typename T>`。项目支持 `Float64` 和 `ComplexF64`。
2.  **子类型化 `<: AbstractTensorTrain`:** 类似于 C++ 的虚基类继承。Julia 利用这一点实现多态分发。
3.  **多指标 (`MultiIndex`):** 在代码中定义为 `Vector{Int}`，代表多维空间的坐标（如 `[2, 5, 1, 3]`）。
4.  **仿函数行为 (`(tt::TensorTrain)(indexset)`):** Julia 允许结构体像函数一样被调用。这在评估插值点时非常频繁。

#### 3. C++ 映射策略建议

针对高性能移植，建议采用以下设计模式：

```cpp
// 建议的 MatrixCI 定义
template <typename T>
struct MatrixCI {
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pivot_cols;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pivot_rows;

    // 映射 Julia 的 evaluate(ci, i, j)
    T evaluate(int i, int j) const {
        // 实现逻辑：left_matrix[i, :] * pivot_rows[:, j]
    }
};

// 建议的 TensorTrain 定义
template <typename T>
class TensorTrain {
public:
    // 使用 std::vector 存储每一层的核心张量
    // 3D 张量推荐使用 Eigen::Tensor<T, 3> 或 xtensor
    std::vector<Eigen::Tensor<T, 3>> site_tensors;

    // 重载调用运算符，映射 Julia 的 tt(indexset)
    T operator()(const std::vector<int>& index_set) const;
};
```

#### 4. 内存管理注意事项

*   **动态增长:** Julia 中的 `push!` 操作在 C++ 中对应 `std::vector::push_back`。但在处理大型核心张量（`sitetensors`）时，频繁的 `vcat`/`hcat` 会导致严重的内存拷贝开销。
*   **预分配建议:** 在 C++ 实现中，如果已知最大秩（`maxbonddim`），建议预先分配稍大的 `Eigen::Matrix` 空间，或者使用 `ConservativeResize` 减少拷贝。
*   **列优先 (Column-Major):** Julia 是列优先存储（同 Fortran）。如果 C++ 侧使用 Eigen（默认列优先），可以直接对接；如果使用 OpenCV 或自定义库，需注意内存布局转换。
