# 文档 2: 数据结构详解 (Data Structures Deep Dive)

## 概述
本文档深入解析 `TensorCrossInterpolation.jl` 中最核心的 5 个数据结构，提供字段级分析、Julia 特性解码以及 C++ 移植策略。

---

## 核心数据结构总览

| 结构名称 | 定义位置 | 类型 | 角色 |
|---------|---------|-----|-----|
| `IndexSet{T}` | `indexset.jl:1` | `struct` | 索引双向映射容器 |
| `MatrixCI{T}` | `matrixci.jl:36` | `mutable struct` | 矩阵交叉插值状态 |
| `rrLU{T}` | `matrixlu.jl:70` | `mutable struct` | 秩揭示LU分解 |
| `TensorCI1{ValueType}` | `tensorci1.jl:5` | `mutable struct` | TCI1算法状态机 |
| `TensorCI2{ValueType}` | `tensorci2.jl:5` | `mutable struct` | TCI2算法状态机 |

---

## 1. `IndexSet{T}` - 索引管理核心

### 定义（`indexset.jl:1-13`）
```julia
struct IndexSet{T}
    toint::Dict{T, Int}      # 索引 → 位置映射
    fromint::Vector{T}        # 位置 → 索引映射
    
    function IndexSet{T}() where {T}
        return new{T}(Dict{T,Int}(), [])
    end
end
```

### 字段解析

| 字段 | 类型 | 作用 | 不变性 |
|-----|------|-----|-------|
| `toint` | `Dict{T, Int}` | 快速查找：给定索引 → 返回其在 `fromint` 中的位置 | `toint[k] == i` ⇔ `fromint[i] == k` |
| `fromint` | `Vector{T}` | 有序存储所有索引，支持迭代和位置访问 | 长度必须等于 `length(keys(toint))` |

### Julia 特性
- **参数化类型** `{T}`: 支持任意类型的索引（整数、元组、向量）
- **内部构造器**: 确保两个字段同步初始化为空
- **不可变性**: `struct`（非 `mutable`），字段不可重新赋值，但可修改容器内容

### C++ 映射策略

#### 方案 1: 双容器实现
```cpp
template<typename T>
class IndexSet {
private:
    std::unordered_map<T, int> to_int_;      // 哈希表
    std::vector<T> from_int_;                 // 动态数组
    
public:
    // 构造器
    IndexSet() = default;
    
    // 添加新索引
    void push(const T& x) {
        from_int_.push_back(x);
        to_int_[x] = from_int_.size() - 1;
    }
    
    // 正向查询：索引 → 位置
    int pos(const T& index) const {
        return to_int_.at(index);
    }
    
    // 反向查询：位置 → 索引
    const T& operator[](int i) const {
        return from_int_[i];
    }
    
    size_t length() const { return from_int_.size(); }
};
```

#### 方案 2: 使用 Boost.Bimap（更高级）
```cpp
#include <boost/bimap.hpp>

template<typename T>
using IndexSet = boost::bimap<T, int>;
```

### 关键操作复杂度

| 操作 | Julia 代码 | 时间复杂度 |
|-----|-----------|----------|
| 添加元素 | `push!(is, x)` | O(1) 平均 |
| 正向查询 | `pos(is, x)` | O(1) 平均 |
| 反向查询 | `is[i]` | O(1) |
| 迭代 | `for x in is` | O(n) |

---

## 2. `MatrixCI{T}` - 矩阵交叉插值

### 定义（`matrixci.jl:36-65`）
```julia
mutable struct MatrixCI{T} <: AbstractMatrixCI{T}
    rowindices::Vector{Int}      # 枢轴行索引（𝓘集合）
    colindices::Vector{Int}       # 枢轴列索引（𝓙集合）
    pivotrows::Matrix{T}          # 枢轴行的完整数据
    pivotcols::Matrix{T}          # 枢轴列的完整数据
    
    function MatrixCI{T}(...) where {T}
        # 验证：size(pivotrows) == (length(rowindices), n_cols_total)
        # 验证：size(pivotcols) == (n_rows_total, length(colindices))
        ...
    end
end
```

### 数学背景
交叉插值将矩阵 **A** (m×n) 分解为：
```
A ≈ A[:, 𝓙] * inv(A[𝓘, 𝓙]) * A[𝓘, :]
```
其中：
- `𝓘 = rowindices`：关键行索引
- `𝓙 = colindices`：关键列索引

### 字段详解

| 字段 | 维度 | 存储内容 | 用途 |
|-----|------|---------|-----|
| `rowindices` | `(r,)` | 已选枢轴行的原始索引 | 标识关键行 |
| `colindices` | `(r,)` | 已选枢轴列的原始索引 | 标识关键列 |
| `pivotrows` | `(r, n)` | 矩阵 A 的枢轴行 | 右因子 |
| `pivotcols` | `(m, r)` | 矩阵 A 的枢轴列 | 左因子 |

**注意**: `r` 是秩（枢轴数），可能 `r ≪ min(m, n)`

### 关键方法

#### `leftmatrix(ci)` - 左矩阵因子
```julia
function leftmatrix(ci::MatrixCI{T}) where {T}
    return AtimesBinv(ci.pivotcols, pivotmatrix(ci))
end
```
计算 `A[:, 𝓙] * inv(A[𝓘, 𝓙])`，使用 QR 分解保证数值稳定性。

#### `AtimesBinv(A, B)` - 核心数值算法
```julia
function AtimesBinv(A::Matrix, B::Matrix)
    # 计算 A * B^{-1}
    # 实现：使用 QR(B') 避免直接求逆
    qr_decomp = qr(transpose(B))
    return transpose(qr_decomp.Q * (qr_decomp.R \ transpose(A)))
end
```

### C++ 映射

```cpp
template<typename T>
class MatrixCI {
private:
    std::vector<int> row_indices_;
    std::vector<int> col_indices_;
    Eigen::Matrix<T, -1, -1> pivot_rows_;  // 动态矩阵
    Eigen::Matrix<T, -1, -1> pivot_cols_;
    
public:
    // 构造器
    MatrixCI(int m, int n) {
        // 初始化为空
    }
    
    // 添加枢轴
    void add_pivot(const Eigen::Matrix<T, -1, -1>& A, int row, int col) {
        // 1. 扩展 row_indices_, col_indices_
        row_indices_.push_back(row);
        col_indices_.push_back(col);
        
        // 2. 扩展 pivot_rows_, pivot_cols_ 矩阵
        // ... (需要动态调整大小并追加行/列)
    }
    
    // 计算左矩阵因子
    Eigen::Matrix<T, -1, -1> left_matrix() const {
        // 等价于 A[:, J] * inv(A[I, J])
        auto pivot_mat = pivot_matrix();
        return pivot_cols_ * pivot_mat.inverse();  // 生产代码需用QR
    }
    
private:
    Eigen::Matrix<T, -1, -1> pivot_matrix() const {
        // 提取 pivotcols[rowindices, :]
        Eigen::Matrix<T, -1, -1> P(row_indices_.size(), col_indices_.size());
        for (int i = 0; i < row_indices_.size(); ++i) {
            P.row(i) = pivot_cols_.row(row_indices_[i]);
        }
        return P;
    }
};
```

**关键挑战**:
- Julia 的矩阵可动态增长（`vcat`, `hcat`），C++ 需要手动管理内存重分配
- 建议使用 Eigen 的 `conservativeResize()` 保留原有数据

---

## 3. `rrLU{T}` - 秩揭示 LU 分解

### 定义（`matrixlu.jl:70-92`）
```julia
mutable struct rrLU{T}
    rowpermutation::Vector{Int}      # 行置换向量
    colpermutation::Vector{Int}      # 列置换向量
    L::Matrix{T}                     # 下三角矩阵
    U::Matrix{T}                     # 上三角矩阵
    leftorthogonal::Bool             # true: L正交, false: U正交
    npivot::Int                      # 已计算的枢轴数（有效秩）
    error::Float64                   # 最后一个枢轴的误差
end
```

### 算法原理
带列主元的 LU 分解：
```
P * A * Q = L * U
```
其中：
- **P**, **Q**: 行/列置换矩阵
- **L**: 下三角（或左正交）
- **U**: 上三角（或右正交）

### 字段详解

| 字段 | 作用 | 更新时机 |
|-----|------|---------|
| `rowpermutation` | 记录行交换历史：原矩阵第 `i` 行 → 分解后第 `rowpermutation[i]` 行 | 每次行交换 |
| `colpermutation` | 记录列交换历史 | 每次列交换 |
| `L` | 下三角因子，维度 `(m, r)` | 添加枢轴时扩展 |
| `U` | 上三角因子，维度 `(r, n)` | 添加枢轴时扩展 |
| `leftorthogonal` | 决定谁是正交矩阵：<br>· `true` → L 列正交<br>· `false` → U 行正交 | 初始化时设定 |
| `npivot` | 当前秩 `r` | 每次成功添加枢轴后 `+1` |
| `error` | `abs(A[pivot_row, pivot_col])` | 每次枢轴选择后更新 |

### 秩截断逻辑
```julia
function _optimizerrlu!(lu::rrLU{T}, A::AbstractMatrix{T}; reltol=1e-14, abstol=0.0)
    while lu.npivot < maxrank
        newerror = abs(A[newpivot])
        
        # 停止条件
        if newerror < max(abstol, reltol * maximum(abs, A))
            break
        end
        
        # 添加枢轴...
        lu.npivot += 1
    end
end
```

### C++ 映射

```cpp
template<typename T>
class RankRevealingLU {
public:
    std::vector<int> row_perm, col_perm;
    Eigen::Matrix<T, -1, -1> L, U;
    bool left_orthogonal;
    int n_pivot;
    double error;
    
    // 构造器：初始化单位置换
    RankRevealingLU(int m, int n, bool left_ortho = true)
        : left_orthogonal(left_ortho), n_pivot(0), error(INFINITY) {
        row_perm.resize(m);
        col_perm.resize(n);
        std::iota(row_perm.begin(), row_perm.end(), 0);
        std::iota(col_perm.begin(), col_perm.end(), 0);
    }
    
    // 主分解函数
    void decompose(Eigen::Matrix<T, -1, -1>& A, 
                   double reltol = 1e-14, 
                   double abstol = 0.0,
                   int max_rank = INT_MAX) {
        while (n_pivot < max_rank) {
            // 1. 在 A[n_pivot:end, n_pivot:end] 中找最大元素
            auto [row, col] = find_max_pivot(A, n_pivot);
            error = std::abs(A(row, col));
            
            // 2. 误差检查
            double threshold = std::max(abstol, reltol * A.array().abs().maxCoeff());
            if (error < threshold) break;
            
            // 3. 行列交换
            swap_row(A, n_pivot, row);
            swap_col(A, n_pivot, col);
            
            // 4. 消元（Schur补）
            update_schur_complement(A, n_pivot);
            
            n_pivot++;
        }
        
        // 提取 L, U
        extract_factors(A);
    }
    
private:
    void swap_row(Eigen::Matrix<T, -1, -1>& A, int i, int j) {
        A.row(i).swap(A.row(j));
        std::swap(row_perm[i], row_perm[j]);
    }
    
    void update_schur_complement(Eigen::Matrix<T, -1, -1>& A, int k) {
        // A[k+1:end, k+1:end] -= A[k+1:end, k] * A[k, k+1:end] / A[k, k]
        T pivot = A(k, k);
        A.block(k+1, k+1, A.rows()-k-1, A.cols()-k-1) -=
            A.block(k+1, k, A.rows()-k-1, 1) * A.block(k, k+1, 1, A.cols()-k-1) / pivot;
    }
};
```

**性能优化**:
- 使用 **Eigen::Block** 避免临时矩阵分配
- **BLAS Level 3** 操作（`gemm`）加速大矩阵

---

## 4. `TensorCI1{ValueType}` - TCI1 算法状态

### 定义（`tensorci1.jl:5-56`）
```julia
mutable struct TensorCI1{ValueType} <: AbstractTensorTrain{ValueType}
    Iset::Vector{IndexSet{MultiIndex}}      # 左索引集（每个键一个）
    Jset::Vector{IndexSet{MultiIndex}}      # 右索引集
    localdims::Vector{Int}                   # 局部物理维度
    
    T::Vector{Matrix{ValueType}}             # 张量核心（2D矩阵形式）
    P::Vector{Matrix{ValueType}}             # 投影矩阵 P_p
    aca::Vector{MatrixCI{ValueType}}         # ACA低秩分解
    
    Pi::Vector{Matrix{ValueType}}            # 辅助矩阵（子采样）
    PiIset::Vector{IndexSet{MultiIndex}}    # Pi对应的左索引
    PiJset::Vector{IndexSet{MultiIndex}}    # Pi对应的右索引
    
    pivoterrors::Vector{Float64}             # 每个键的枢轴误差
    maxsamplevalue::Float64                  # 全局最大采样值
end
```

### 数据流关系图
```
函数 f(x) → 采样 → Pi 矩阵 
               ↓
           交叉插值 → aca → T, P 矩阵
               ↓
          更新 Iset, Jset
```

### 关键不变量
1. **维度一致性**:
   ```julia
   length(Iset) == length(Jset) == length(T) == length(localdims)
   ```
2. **索引集与张量核心的对齐**:
   ```julia
   size(T[p]) == (length(Iset[p]), localdims[p] * length(Jset[p]))
   ```
3. **ACA 与 T 的同步**:
   ```julia
   leftmatrix(aca[p]) * pivotrows(aca[p]) ≈ T[p]
   ```

### C++ 映射

```cpp
template<typename T>
class TensorCI1 {
public:
    using MultiIndex = std::vector<int>;
    
    std::vector<IndexSet<MultiIndex>> I_set;  // 左索引集
    std::vector<IndexSet<MultiIndex>> J_set;  // 右索引集
    std::vector<int> local_dims;
    
    std::vector<Eigen::Matrix<T, -1, -1>> site_tensors;  // T 核心
    std::vector<Eigen::Matrix<T, -1, -1>> projectors;    // P 矩阵
    std::vector<MatrixCI<T>> aca_decomposes;
    
    std::vector<Eigen::Matrix<T, -1, -1>> Pi;
    std::vector<IndexSet<MultiIndex>> Pi_I_set;
    std::vector<IndexSet<MultiIndex>> Pi_J_set;
    
    std::vector<double> pivot_errors;
    double max_sample_value;
    
    // 构造器
    TensorCI1(const std::vector<int>& dims) 
        : local_dims(dims), max_sample_value(0.0) {
        int n = dims.size();
        I_set.resize(n);
        J_set.resize(n);
        // ... 初始化其他容器
    }
    
    // 核心方法
    template<typename Func>
    void add_global_pivot(Func& f, const MultiIndex& pivot, double tolerance);
    
    T evaluate(const MultiIndex& index) const;
    
    double last_sweep_error() const {
        return *std::max_element(pivot_errors.begin(), pivot_errors.end());
    }
};
```

**内存管理**:
- Julia 依赖 GC，C++ 需显式管理 `std::vector` 的扩容
- 使用 `reserve()` 预分配减少重分配

---

## 5. `TensorCI2{ValueType}` - TCI2 算法状态

### 定义（`tensorci2.jl:5-40`）
```julia
mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}
    Iset::Vector{Vector{MultiIndex}}         # 嵌套索引集（支持多轮扫描）
    Jset::Vector{Vector{MultiIndex}}
    localdims::Vector{Int}
    
    sitetensors::Vector{Array{ValueType, 3}} # 3D站点张量
    
    bonderrors::Vector{Float64}               # 键误差（比TCI1多一层）
    pivoterrors::Vector{Vector{Float64}}      # 嵌套枢轴误差
    
    maxsamplevalue::Float64
    Iset_history::Vector{Vector{Vector{MultiIndex}}}  # 嵌套历史
    Jset_history::Vector{Vector{Vector{MultiIndex}}}
end
```

### TCI1 vs TCI2 对比

| 特性 | TCI1 | TCI2 |
|-----|------|------|
| 索引集类型 | `IndexSet{MultiIndex}` | `Vector{MultiIndex}` |
| 扫描策略 | 单向（half-sweep） | 双向（full-sweep + 2site） |
| 张量存储 | 2D 矩阵 | 3D 数组 |
| 误差跟踪 | 简单向量 | 层次化嵌套 |
| 历史记录 | 无 | 完整历史（用于嵌套性检查） |

### 3D 站点张量结构
```julia
sitetensors[p]::Array{ValueType, 3}
```
维度布局：
```
(left_bond_dim, physical_dim, right_bond_dim)
    ↓                ↓               ↓
length(Iset[p]), localdims[p], length(Jset[p])
```

### C++ 映射

```cpp
template<typename T>
class TensorCI2 {
public:
    using MultiIndex = std::vector<int>;
    
    // 注意：双重嵌套
    std::vector<std::vector<MultiIndex>> I_set;
    std::vector<std::vector<MultiIndex>> J_set;
    std::vector<int> local_dims;
    
    // 使用 Eigen::Tensor（需要 unsupported 模块）或自定义3D数组
    std::vector<Eigen::Tensor<T, 3>> site_tensors;
    
    std::vector<double> bond_errors;
    std::vector<std::vector<double>> pivot_errors;
    
    double max_sample_value;
    
    // 历史记录
    std::vector<std::vector<std::vector<MultiIndex>>> I_set_history;
    std::vector<std::vector<std::vector<MultiIndex>>> J_set_history;
    
    // ... 方法实现
};
```

**关键差异**:
- TCI2 需要支持 **3D 张量** 操作，Eigen 的 `Tensor` 模块性能不如矩阵模块
- 可选方案：将 3D 张量展平为 2D（牺牲可读性换性能）

---

## 额外结构：`CachedFunction{ValueType, K}`

### 定义（`cachedfunction.jl:7-30`）
```julia
struct CachedFunction{ValueType, K<:Union{UInt32,...,BigInt}}
    f::Function                    # 原函数
    localdims::Vector{Int}         # 维度信息
    cache::Dict{K, ValueType}      # 键值缓存
    coeffs::Vector{K}              # 哈希系数向量
end
```

### 哈希策略
将多索引 `[i1, i2, ..., iN]` 映射为单个整数键：
```julia
function _key(cf, indexset::Vector{Int})::K
    result = zero(K)
    for i in 1:N
        result += coeffs[i] * (indexset[i] - 1)
    end
    return result
end
```

**C++ 实现**:
```cpp
template<typename T, typename KeyType = uint64_t>
class CachedFunction {
    std::function<T(const std::vector<int>&)> f_;
    std::vector<int> local_dims_;
    std::unordered_map<KeyType, T> cache_;
    std::vector<KeyType> coeffs_;
    
public:
    T operator()(const std::vector<int>& index) {
        KeyType key = compute_key(index);
        
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;  // 缓存命中
        }
        
        T val = f_(index);
        cache_[key] = val;
        return val;
    }
    
private:
    KeyType compute_key(const std::vector<int>& idx) const {
        KeyType k = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            k += coeffs_[i] * (idx[i] - 1);
        }
        return k;
    }
};
```

**溢出检测**:
Julia 代码检查键不溢出：
```julia
sum(coeffs .* (localdims .- 1)) < typemax(K)
```
C++ 需手动验证或使用大整数类型。

---

## 总结：C++ 移植难度评估

| 结构 | 难度 | 关键挑战 |
|-----|------|---------|
| `IndexSet` | ⭐ | 简单，直接用 `std::unordered_map` + `std::vector` |
| `MatrixCI` | ⭐⭐ | 需要 Eigen，动态矩阵扩展 |
| `rrLU` | ⭐⭐⭐ | 数值稳定性，置换管理 |
| `TensorCI1` | ⭐⭐⭐⭐ | 复杂状态管理，多矩阵同步 |
| `TensorCI2` | ⭐⭐⭐⭐⭐ | 3D张量 + 嵌套历史，最复杂 |
| `CachedFunction` | ⭐⭐ | 哈希函数设计，溢出检查 |

**建议实现顺序**:
1. `IndexSet` → 2. `CachedFunction` → 3. `MatrixCI` → 4. `rrLU` → 5. `TensorCI1` → 6. `TensorCI2`
