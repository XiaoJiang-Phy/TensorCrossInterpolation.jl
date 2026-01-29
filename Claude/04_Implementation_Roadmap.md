# 文档 4: 移植与研读计划 (Implementation Roadmap)

## 概述
本文档为 C++ 工程师提供从零开始移植 `TensorCrossInterpolation.jl` 的详细路线图，包括模块拆解、难点预警和实施优先级。

---

## 第一部分：模块拆解（6 个独立模块）

### 模块 1: 基础工具 (Foundation)
**优先级**: ⭐⭐⭐⭐⭐ (最高)  
**难度**: ★☆☆☆☆

#### 包含文件
- `util.jl` → `util.hpp/cpp`
- `indexset.jl` → `indexset.hpp/cpp`

#### 核心功能
1. **`IndexSet<T>` 类**:
   ```cpp
   template<typename T>
   class IndexSet {
   private:
       std::unordered_map<T, int> to_int_;
       std::vector<T> from_int_;
       
   public:
       void push(const T& item);
       int pos(const T& item) const;
       const T& operator[](int i) const;
       size_t length() const;
       
       // 迭代器支持
       auto begin() { return from_int_.begin(); }
       auto end() { return from_int_.end(); }
   };
   
   // 特化类型
   using MultiIndex = std::vector<int>;
   using IndexSetMI = IndexSet<MultiIndex>;
   ```

2. **工具函数**:
   ```cpp
   // util.hpp
   double max_abs(double current, const Eigen::MatrixXd& updates);
   
   template<typename T>
   void push_unique(std::vector<T>& collection, const T& item);
   
   std::vector<int> opt_first_pivot(
       std::function<double(const MultiIndex&)> f,
       const std::vector<int>& local_dims,
       const MultiIndex& initial_pivot = {},
       int max_sweep = 1000
   );
   ```

#### 实现建议
- **哈希函数**: 为 `MultiIndex` 定义自定义哈希
  ```cpp
  namespace std {
  template<>
  struct hash<MultiIndex> {
      size_t operator()(const MultiIndex& v) const {
          size_t seed = 0;
          for (int x : v) {
              seed ^= std::hash<int>{}(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
          }
          return seed;
      }
  };
  }
  ```

#### 测试用例
```cpp
// test_indexset.cpp
void test_indexset() {
    IndexSet<int> iset;
    iset.push(5);
    iset.push(10);
    assert(iset.pos(5) == 0);
    assert(iset[1] == 10);
}
```

---

### 模块 2: 线性代数核心 (Linear Algebra Backend)
**优先级**: ⭐⭐⭐⭐⭐  
**难度**: ★★★★☆

#### 包含文件
- `matrixci.jl` → `matrix_ci.hpp/cpp`
- `matrixlu.jl` → `matrix_lu.hpp/cpp`
- `matrixaca.jl` → `matrix_aca.hpp/cpp`

#### 核心数据结构

##### 1. `MatrixCI<T>` - 矩阵交叉插值
```cpp
template<typename T>
class MatrixCI {
private:
    std::vector<int> row_indices_;
    std::vector<int> col_indices_;
    Eigen::Matrix<T, -1, -1> pivot_rows_;
    Eigen::Matrix<T, -1, -1> pivot_cols_;
    
public:
    MatrixCI(int m, int n);
    
    void add_pivot(const Eigen::Matrix<T, -1, -1>& A, int row, int col);
    
    Eigen::Matrix<T, -1, -1> left_matrix() const;
    Eigen::Matrix<T, -1, -1> right_matrix() const;
    
    std::pair<double, std::pair<int,int>> find_new_pivot(
        const Eigen::Matrix<T, -1, -1>& A
    ) const;
};
```

##### 2. `RankRevealingLU<T>` - 秩揭示 LU 分解
```cpp
template<typename T>
class RankRevealingLU {
public:
    std::vector<int> row_perm, col_perm;
    Eigen::Matrix<T, -1, -1> L, U;
    bool left_orthogonal;
    int n_pivot;
    double error;
    
    RankRevealingLU(int m, int n, bool left_ortho = true);
    
    void decompose(
        Eigen::Matrix<T, -1, -1>& A,
        double reltol = 1e-14,
        double abstol = 0.0,
        int max_rank = INT_MAX
    );
    
    const std::vector<int>& row_indices() const { return row_perm; }
    const std::vector<int>& col_indices() const { return col_perm; }
    double last_pivot_error() const { return error; }
    
private:
    void swap_row(Eigen::Matrix<T, -1, -1>& A, int i, int j);
    void swap_col(Eigen::Matrix<T, -1, -1>& A, int i, int j);
    void update_schur_complement(Eigen::Matrix<T, -1, -1>& A, int k);
    std::pair<int,int> find_max_pivot(const Eigen::Matrix<T, -1, -1>& A, int k);
};
```

#### 关键算法实现

##### `AtimesBinv` (QR 分解求 A*B^{-1})
```cpp
template<typename T>
Eigen::Matrix<T, -1, -1> AtimesBinv(
    const Eigen::Matrix<T, -1, -1>& A,
    const Eigen::Matrix<T, -1, -1>& B
) {
    // 使用 Eigen 的 QR 分解
    Eigen::HouseholderQR<Eigen::Matrix<T, -1, -1>> qr(B.transpose());
    return (qr.solve(A.transpose())).transpose();
}
```

##### Schur 补更新（LU 分解核心）
```cpp
template<typename T>
void RankRevealingLU<T>::update_schur_complement(
    Eigen::Matrix<T, -1, -1>& A, 
    int k
) {
    T pivot = A(k, k);
    if (std::abs(pivot) < 1e-15) {
        throw std::runtime_error("Near-zero pivot encountered");
    }
    
    // A[k+1:end, k+1:end] -= A[k+1:end, k] * A[k, k+1:end] / pivot
    A.block(k+1, k+1, A.rows()-k-1, A.cols()-k-1).noalias() -=
        (A.block(k+1, k, A.rows()-k-1, 1) * 
         A.block(k, k+1, 1, A.cols()-k-1)) / pivot;
}
```

#### 难点预警 ⚠️

**挑战 1: 动态矩阵扩展**
- **Julia**: `vcat(A, new_row)` 自动处理内存
- **C++ 解决方案**:
  ```cpp
  void expand_matrix(Eigen::MatrixXd& M, int new_rows, int new_cols) {
      M.conservativeResize(new_rows, new_cols);
  }
  ```

**挑战 2: 数值稳定性**
- **必须**: 使用 `Eigen::HouseholderQR`（不要用 `.inverse()`）
- **测试**: 对比 Julia 结果，误差应 < 1e-12

---

### 模块 3: 函数缓存 (Function Caching)
**优先级**: ⭐⭐⭐⭐  
**难度**: ★★☆☆☆

#### 包含文件
- `cachedfunction.jl` → `cached_function.hpp/cpp`

#### 核心实现

```cpp
template<typename T, typename KeyType = uint64_t>
class CachedFunction {
private:
    std::function<T(const MultiIndex&)> f_;
    std::vector<int> local_dims_;
    std::unordered_map<KeyType, T> cache_;
    std::vector<KeyType> coeffs_;
    
    KeyType compute_key(const MultiIndex& index) const {
        KeyType key = 0;
        for (size_t i = 0; i < index.size(); ++i) {
            key += coeffs_[i] * (index[i] - 1);
        }
        return key;
    }
    
public:
    CachedFunction(
        std::function<T(const MultiIndex&)> f,
        const std::vector<int>& local_dims
    ) : f_(f), local_dims_(local_dims) {
        // 计算哈希系数
        coeffs_.resize(local_dims.size());
        coeffs_[0] = 1;
        for (size_t i = 1; i < local_dims.size(); ++i) {
            coeffs_[i] = coeffs_[i-1] * local_dims[i-1];
        }
        
        // 溢出检查
        KeyType max_key = 0;
        for (size_t i = 0; i < local_dims.size(); ++i) {
            max_key += coeffs_[i] * (local_dims[i] - 1);
        }
        if (max_key > std::numeric_limits<KeyType>::max() / 2) {
            throw std::overflow_error("Index space too large for key type");
        }
    }
    
    T operator()(const MultiIndex& index) {
        KeyType key = compute_key(index);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;  // 缓存命中
        }
        
        T value = f_(index);
        cache_[key] = value;
        return value;
    }
    
    size_t cache_size() const { return cache_.size(); }
    void clear_cache() { cache_.clear(); }
};
```

#### 大整数支持

对于超高维问题（如 20+ 维度），使用 **Boost.Multiprecision**:
```cpp
#include <boost/multiprecision/cpp_int.hpp>

using BigInt = boost::multiprecision::uint256_t;
using CachedFunctionBig = CachedFunction<double, BigInt>;
```

---

### 模块 4: 张量结构 (Tensor Train Framework)
**优先级**: ⭐⭐⭐⭐  
**难度**: ★★★☆☆

#### 包含文件
- `abstracttensortrain.jl` → `abstract_tensor_train.hpp`
- `tensortrain.jl` → `tensor_train.hpp/cpp`

#### 抽象基类设计

```cpp
template<typename T>
class AbstractTensorTrain {
public:
    virtual ~AbstractTensorTrain() = default;
    
    // 纯虚函数
    virtual int length() const = 0;
    virtual std::vector<std::vector<int>> site_dims() const = 0;
    virtual std::vector<int> link_dims() const = 0;
    
    // 实现的公共方法
    T evaluate(const MultiIndex& index) const;
    T sum() const;
    double norm() const;
    
protected:
    virtual const Eigen::Tensor<T, 3>& site_tensor(int i) const = 0;
};
```

#### 具体实现：`TensorTrain`

```cpp
template<typename T>
class TensorTrain : public AbstractTensorTrain<T> {
private:
    std::vector<Eigen::Tensor<T, 3>> site_tensors_;
    
public:
    // 构造器
    TensorTrain(const std::vector<Eigen::Tensor<T, 3>>& tensors)
        : site_tensors_(tensors) {}
    
    // 从 TensorCI2 转换
    explicit TensorTrain(const TensorCI2<T>& tci);
    
    // 压缩操作
    void compress(
        double tolerance = 1e-12,
        int max_bond_dim = INT_MAX,
        CompressionMethod method = CompressionMethod::SVD
    );
    
    // 张量求值
    T evaluate(const MultiIndex& index) const override {
        // 矩阵乘法链: T[0][:, i0, :] * T[1][:, i1, :] * ...
        Eigen::Matrix<T, 1, -1> result = 
            site_tensors_[0].chip(index[0], 1).reshape(Eigen::Vector2i(1, -1));
        
        for (size_t i = 1; i < site_tensors_.size(); ++i) {
            Eigen::Matrix<T, -1, -1> slice = 
                site_tensors_[i].chip(index[i], 1);
            result = result * slice;
        }
        return result(0, 0);
    }
    
    // 张量求和
    T sum() const override;
    
    int length() const override { return site_tensors_.size(); }
    
protected:
    const Eigen::Tensor<T, 3>& site_tensor(int i) const override {
        return site_tensors_[i];
    }
};
```

#### 3D 张量操作难点 ⚠️

**问题**: Eigen 的 `Tensor` 模块不如 `Matrix` 成熟

**解决方案 1**: 使用 Eigen::Tensor (unsupported)
```cpp
#include <unsupported/Eigen/CXX11/Tensor>

Eigen::Tensor<double, 3> T(2, 3, 4);  // (left_dim, phys_dim, right_dim)
auto slice = T.chip(1, 1);  // 固定物理索引为1
```

**解决方案 2**: 展平为 2D 矩阵（推荐）
```cpp
// 将 3D 张量 (l, d, r) 存储为 2D 矩阵 (l*d, r)
struct SiteTensor {
    int left_dim, phys_dim, right_dim;
    Eigen::MatrixXd data;  // 形状: (left_dim * phys_dim, right_dim)
    
    Eigen::MatrixXd get_slice(int phys_index) const {
        return data.block(
            left_dim * phys_index, 0, 
            left_dim, right_dim
        );
    }
};
```

---

### 模块 5: TCI1 算法 (First-Generation Cross Interpolation)
**优先级**: ⭐⭐⭐  
**难度**: ★★★★☆

#### 包含文件
- `tensorci1.jl` → `tensor_ci1.hpp/cpp`

#### 核心结构

```cpp
template<typename T>
class TensorCI1 : public AbstractTensorTrain<T> {
private:
    std::vector<IndexSetMI> I_set_;
    std::vector<IndexSetMI> J_set_;
    std::vector<int> local_dims_;
    
    std::vector<Eigen::Matrix<T, -1, -1>> T_cores_;
    std::vector<Eigen::Matrix<T, -1, -1>> P_projectors_;
    std::vector<MatrixCI<T>> aca_;
    
    std::vector<Eigen::Matrix<T, -1, -1>> Pi_;
    std::vector<IndexSetMI> Pi_I_set_;
    std::vector<IndexSetMI> Pi_J_set_;
    
    std::vector<double> pivot_errors_;
    double max_sample_value_;
    
public:
    TensorCI1(const std::vector<int>& local_dims);
    
    template<typename Func>
    void add_global_pivot(
        Func& f, 
        const MultiIndex& pivot, 
        double tolerance
    );
    
    double last_sweep_pivot_error() const {
        return *std::max_element(pivot_errors_.begin(), pivot_errors_.end());
    }
    
    // ... 其他方法
};
```

**建议**: 先实现 TCI2，TCI1 可作为简化版参考

---

### 模块 6: TCI2 算法 (Second-Generation Cross Interpolation)
**优先级**: ⭐⭐⭐⭐⭐  
**难度**: ★★★★★ (最高)

#### 包含文件
- `tensorci2.jl` → `tensor_ci2.hpp/cpp`
- `globalpivotfinder.jl` → `global_pivot_finder.hpp/cpp`
- `globalsearch.jl` → `global_search.hpp/cpp`

#### 核心结构

```cpp
template<typename T>
class TensorCI2 : public AbstractTensorTrain<T> {
private:
    std::vector<std::vector<MultiIndex>> I_set_;  // 注意双重嵌套
    std::vector<std::vector<MultiIndex>> J_set_;
    std::vector<int> local_dims_;
    
    std::vector<Eigen::Tensor<T, 3>> site_tensors_;
    
    std::vector<double> bond_errors_;
    std::vector<std::vector<double>> pivot_errors_;
    double max_sample_value_;
    
    // 历史记录（用于嵌套性检查）
    std::vector<std::vector<std::vector<MultiIndex>>> I_set_history_;
    std::vector<std::vector<std::vector<MultiIndex>>> J_set_history_;
    
public:
    TensorCI2(const std::vector<int>& local_dims);
    
    template<typename Func>
    TensorCI2(
        Func& f,
        const std::vector<int>& local_dims,
        const std::vector<MultiIndex>& initial_pivots = {}
    );
    
    // 主优化函数
    template<typename Func>
    std::pair<std::vector<int>, std::vector<double>> optimize(
        Func& f,
        double tolerance = 1e-8,
        int max_bond_dim = INT_MAX,
        int max_iter = 20,
        SweepStrategy strategy = SweepStrategy::BackAndForth,
        PivotSearch search = PivotSearch::Full,
        int verbosity = 0,
        bool normalize_error = true
    );
    
private:
    template<typename Func>
    void sweep_2site(
        Func& f, 
        int n_iter,
        double abstol,
        int max_bond_dim,
        SweepStrategy strategy,
        PivotSearch search,
        int verbosity
    );
    
    template<typename Func>
    void update_pivots(
        Func& f,
        int bond_index,
        bool left_orthogonal,
        double abstol,
        int max_bond_dim,
        PivotSearch search,
        int verbosity
    );
};
```

#### 辅助枚举

```cpp
enum class SweepStrategy {
    Forward,
    Backward,
    BackAndForth
};

enum class PivotSearch {
    Full,   // 全矩阵采样
    Rook    // 稀疏 Rook 策略
};
```

#### 全局枢轴搜索接口

```cpp
class AbstractGlobalPivotFinder {
public:
    virtual ~AbstractGlobalPivotFinder() = default;
    
    virtual std::vector<MultiIndex> operator()(
        const TensorCI2<double>& tci,
        std::function<double(const MultiIndex&)> f,
        double abstol,
        int verbosity = 0
    ) = 0;
};

class DefaultGlobalPivotFinder : public AbstractGlobalPivotFinder {
private:
    int nsearch_;
    int max_n_global_pivot_;
    double tol_margin_;
    
public:
    DefaultGlobalPivotFinder(
        int nsearch = 5,
        int max_n_global_pivot = 5,
        double tol_margin = 10.0
    );
    
    std::vector<MultiIndex> operator()(
        const TensorCI2<double>& tci,
        std::function<double(const MultiIndex&)> f,
        double abstol,
        int verbosity
    ) override;
};
```

---

## 第二部分：难点预警与解决方案

### 难点 1: Julia 的动态类型系统

**问题**: Julia 允许混合类型操作
```julia
T = Vector{Any}()  # 可以存储任意类型
push!(T, Matrix{Float64}(...))
push!(T, ComplexF64(...))
```

**C++ 解决方案**: 使用 `std::variant` 或模板特化
```cpp
// 方案1: 模板参数化所有类型
template<typename T>
class TensorCI2 { /* ... */ };

// 方案2: 有限类型集合使用 variant
using TensorValue = std::variant<double, std::complex<double>>;
```

**建议**: 初期只支持 `double` 和 `std::complex<double>`

---

### 难点 2: 广播操作 (Broadcasting)

**Julia 代码**:
```julia
A .+ B  # 自动扩展维度
f.(x)   # 对 x 中每个元素应用 f
```

**C++ 解决方案**: Eigen 的 `.array()` 模式
```cpp
// Julia: C = A .+ B
Eigen::MatrixXd C = A.array() + B.array();

// Julia: result = abs.(A)
auto result = A.array().abs();

// 自定义函数: f.(x)
auto result = x.unaryExpr([](double v) { return std::sin(v); });
```

---

### 难点 3: 宏系统

**Julia 宏**:
```julia
@doc raw"""..."""  # 文档字符串
@inbounds A[i]     # 禁用边界检查
@fastmath x + y    # 快速数学模式
```

**C++ 对应**:
```cpp
// @doc → Doxygen 注释
/**
 * @brief Matrix cross interpolation
 * @param A Input matrix
 */

// @inbounds → 编译器提示
#ifndef NDEBUG
    assert(i < size);
#endif

// @fastmath → 编译器标志
// GCC/Clang: -ffast-math
// MSVC: /fp:fast
```

---

### 难点 4: 多重分派 (Multiple Dispatch)

**Julia 示例**:
```julia
function evaluate(tci::TensorCI1, index::Vector{Int})
    # 实现1
end

function evaluate(tci::TensorCI2, index::Vector{Int})
    # 实现2
end
```

**C++ 解决方案**: 虚函数 + 重载
```cpp
class AbstractTensorTrain {
public:
    virtual double evaluate(const MultiIndex& index) const = 0;
};

class TensorCI1 : public AbstractTensorTrain {
public:
    double evaluate(const MultiIndex& index) const override {
        // 实现1
    }
};

class TensorCI2 : public AbstractTensorTrain {
public:
    double evaluate(const MultiIndex& index) const override {
        // 实现2
    }
};
```

---

### 难点 5: 内存管理

**Julia**: 自动垃圾回收
```julia
A = rand(1000, 1000)  # 不需要手动释放
```

**C++ 策略**:
```cpp
// 方案1: RAII (推荐)
class TensorCI2 {
private:
    std::vector<Eigen::Tensor<double, 3>> tensors_;  // 自动析构
};

// 方案2: 智能指针（用于多态）
std::unique_ptr<AbstractTensorTrain> tci = 
    std::make_unique<TensorCI2<double>>(...);

// 避免裸指针！
```

**内存优化**:
```cpp
// 预分配容器大小
std::vector<Eigen::MatrixXd> tensors;
tensors.reserve(n_sites);

// 移动语义避免拷贝
Eigen::MatrixXd compute_matrix() {
    Eigen::MatrixXd result = ...;
    return result;  // 返回值优化 (RVO)
}
```

---

### 难点 6: 函数式编程

**Julia 高阶函数**:
```julia
result = map(f, collection)
filtered = filter(x -> x > 0, collection)
```

**C++ 解决方案**: STL 算法 + Lambda
```cpp
#include <algorithm>

// map
std::vector<double> result;
std::transform(collection.begin(), collection.end(), 
               std::back_inserter(result),
               [](auto x) { return f(x); });

// filter
std::vector<int> filtered;
std::copy_if(collection.begin(), collection.end(),
             std::back_inserter(filtered),
             [](int x) { return x > 0; });
```

---

## 第三部分：实施优先级与时间估算

### 阶段 1: 基础设施 (2-3 周)
- [ ] 模块 1: 基础工具 (3 天)
  - `util.hpp/cpp`
  - `indexset.hpp/cpp`
  - 单元测试
- [ ] 模块 2: 线性代数 (1.5 周)
  - `matrix_ci.hpp/cpp`
  - `matrix_lu.hpp/cpp`
  - 集成 Eigen/LAPACK
  - **里程碑**: 矩阵交叉插值示例运行成功
- [ ] 模块 3: 函数缓存 (2 天)
  - `cached_function.hpp/cpp`
  - 哈希函数优化

### 阶段 2: 核心算法 (4-6 周)
- [ ] 模块 4: 张量列框架 (1 周)
  - `tensor_train.hpp/cpp`
  - 求值、求和、压缩
- [ ] 模块 6: TCI2 算法 (3-4 周)
  - `tensor_ci2.hpp/cpp`
  - `sweep2site` 实现
  - `optimize` 主循环
  - **里程碑**: 复现 Julia 的 Lorentzian 示例结果
- [ ] 模块 6.5: 全局搜索 (1 周)
  - `global_pivot_finder.hpp/cpp`
  - 随机搜索策略

### 阶段 3: 优化与验证 (2-4 周)
- [ ] 性能优化 (1 周)
  - 批量求值优化
  - BLAS/LAPACK 调用优化
  - 多线程支持（OpenMP）
- [ ] 数值测试 (1 周)
  - 与 Julia 结果对比（误差 < 1e-10）
  - 大规模问题测试（10+ 维度）
- [ ] 文档与示例 (1-2 周)
  - Doxygen 文档生成
  - 教程示例
  - API 参考手册

### 总计时间: **8-13 周**

---

## 第四部分：测试驱动开发建议

### 单元测试框架

**推荐**: Google Test
```cpp
// test_matrix_ci.cpp
#include <gtest/gtest.h>
#include "matrix_ci.hpp"

TEST(MatrixCITest, BasicInterpolation) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 10);
    MatrixCI<double> ci(10, 10);
    
    ci.add_pivot(A, 0, 0);
    ci.add_pivot(A, 5, 5);
    
    auto [error, pivot] = ci.find_new_pivot(A);
    EXPECT_LT(error, 1.0);  // 误差应下降
}
```

### 对比测试（与 Julia 结果）

```cpp
// test_against_julia.cpp
TEST(TensorCI2Test, LorentzianFunction) {
    // 定义函数: f(v) = 1 / (1 + v'v)
    auto f = [](const MultiIndex& v) {
        double sum_sq = 0.0;
        for (int x : v) sum_sq += x * x;
        return 1.0 / (1.0 + sum_sq);
    };
    
    // 设置参数（与 Julia 测试一致）
    std::vector<int> local_dims = {10, 10, 10, 10};
    TensorCI2<double> tci(f, local_dims);
    
    auto [ranks, errors] = tci.optimize(
        f, 
        /*tolerance=*/1e-8,
        /*max_bond_dim=*/100
    );
    
    // 从 Julia 保存的参考结果加载
    auto julia_ranks = load_reference("julia_lorentz_ranks.txt");
    auto julia_errors = load_reference("julia_lorentz_errors.txt");
    
    // 验证秩相同
    ASSERT_EQ(ranks.size(), julia_ranks.size());
    for (size_t i = 0; i < ranks.size(); ++i) {
        EXPECT_EQ(ranks[i], julia_ranks[i]);
    }
    
    // 验证误差接近
    for (size_t i = 0; i < errors.size(); ++i) {
        EXPECT_NEAR(errors[i], julia_errors[i], 1e-10);
    }
}
```

---

## 第五部分：工具链建议

### 编译器
- **Linux**: GCC 11+ 或 Clang 14+
- **Windows**: MSVC 2022 或 MinGW-w64
- **macOS**: Apple Clang 13+

### 依赖库

| 库 | 版本 | 用途 | 安装命令 |
|---|------|------|---------|
| **Eigen** | 3.4+ | 线性代数核心 | `sudo apt install libeigen3-dev` |
| **LAPACK** | 3.9+ | 高性能矩阵分解 | `sudo apt install liblapack-dev` |
| **Google Test** | 1.12+ | 单元测试 | `sudo apt install libgtest-dev` |
| **Boost** (可选) | 1.75+ | 大整数、Bimap | `sudo apt install libboost-all-dev` |

### 构建系统

**推荐**: CMake
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(TensorCrossInterpolationCPP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(Eigen3 3.4 REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)

# 添加库
add_library(tci_core
    src/util.cpp
    src/indexset.cpp
    src/matrix_ci.cpp
    src/matrix_lu.cpp
    src/tensor_train.cpp
    src/tensor_ci2.cpp
)

target_include_directories(tci_core PUBLIC include)
target_link_libraries(tci_core PUBLIC Eigen3::Eigen ${LAPACK_LIBRARIES})

# 添加测试
enable_testing()
add_executable(test_tci tests/test_all.cpp)
target_link_libraries(test_tci tci_core GTest::GTest GTest::Main)
add_test(NAME all_tests COMMAND test_tci)
```

---

## 第六部分：常见陷阱与最佳实践

### 陷阱 1: 列主序 vs 行主序

**Julia**: 列主序（Fortran 风格）
```julia
A[i, j]  # 存储: A[1,1], A[2,1], A[3,1], ...
```

**Eigen 默认**: 列主序（与 Julia 一致）
```cpp
Eigen::MatrixXd A(3, 3);  // 默认列主序
A(i, j);  // OK
```

**陷阱**: 如果使用行主序会导致性能下降！
```cpp
// 避免：
Eigen::Matrix<double, -1, -1, Eigen::RowMajor> A;  // 行主序
```

### 陷阱 2: 1-based vs 0-based 索引

**Julia**: 1-based
```julia
v = [10, 20, 30]
v[1]  # => 10
```

**C++**: 0-based
```cpp
std::vector<int> v = {10, 20, 30};
v[0];  // => 10
```

**解决方案**: 在接口层转换
```cpp
class MultiIndexAdapter {
public:
    // Julia 风格接口（1-indexed）
    static MultiIndex from_julia(const std::vector<int>& julia_index) {
        MultiIndex cpp_index = julia_index;
        for (auto& x : cpp_index) x--;  // 转换为 0-based
        return cpp_index;
    }
};
```

### 最佳实践 1: 使用 `const` 引用避免拷贝

```cpp
// 好：
void process(const Eigen::MatrixXd& A);

// 不好：
void process(Eigen::MatrixXd A);  // 会拷贝整个矩阵！
```

### 最佳实践 2: 移动语义

```cpp
class TensorCI2 {
private:
    std::vector<Eigen::Tensor<double, 3>> tensors_;
    
public:
    // 接受右值（移动）
    void set_tensor(int i, Eigen::Tensor<double, 3>&& T) {
        tensors_[i] = std::move(T);  // 无拷贝
    }
};
```

### 最佳实践 3: 编译时优化

```cmake
# CMakeLists.txt
if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(tci_core PRIVATE
        -O3 -march=native -ffast-math
    )
endif()
```

---

## 结论

### 建议的实施路径
```
Week 1-3:   模块 1 + 模块 2 (基础 + 线性代数)
Week 4-5:   模块 3 + 模块 4 (缓存 + 张量列)
Week 6-10:  模块 6 (TCI2 核心算法)
Week 11-13: 测试、优化、文档
```

### 成功标准
- [ ] 通过所有单元测试
- [ ] Lorentzian 示例结果与 Julia 误差 < 1e-10
- [ ] 8维函数插值性能 < 10秒（单线程）
- [ ] 文档覆盖率 > 80%

### 扩展方向
1. **GPU 加速**: 使用 CUDA 或 Eigen::CUDA
2. **分布式计算**: MPI 并行化
3. **Python 绑定**: pybind11 包装 C++ 库
4. **自动微分**: 集成 Eigen::AutoDiff

**祝移植顺利！**
