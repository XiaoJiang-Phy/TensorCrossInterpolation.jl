# 文档 1: 项目物理结构 (Project Structure)

## 概述
`TensorCrossInterpolation.jl` 是一个实现张量交叉插值（Tensor Cross Interpolation, TCI）算法的 Julia 库。该库用于高效插值多索引张量和多元函数，核心应用于量子计算和高性能科学计算领域。

---

## 1. 文件清单与角色映射

### 主模块入口
- **`TensorCrossInterpolation.jl`** (45行, 1156字节)
  - **角色**: 主模块定义文件，声明模块依赖、导出符号、并按顺序加载所有子模块
  - **关键导出**:
    - `crossinterpolate1`, `crossinterpolate2`: 两种TCI算法的顶层接口
    - `tensortrain`, `TensorTrain`: 张量列（Tensor Train）数据结构
    - `contract`: 张量收缩操作
  - **依赖库**: `LinearAlgebra`, `EllipsisNotation`, `BitIntegers`, `QuadGK`

---

### 核心数据结构 (6个文件)

#### 1. **`abstracttensortrain.jl`** (293行, 10662字节)
- **角色**: 定义抽象基类型 `AbstractTensorTrain{V}`，为所有张量列类型提供统一接口
- **核心职责**:
  - 定义张量列的通用操作：`linkdims` (键维度), `sitedims` (局部维度), `evaluate` (求值)
  - 实现张量列的加法/减法运算（通过 `add`, `subtract`, `+`, `-`）
  - 实现 Frobenius 范数计算 (`LA.norm`, `LA.norm2`)
- **类型层次**: 这是所有TCI类型继承的父类

#### 2. **`tensortrain.jl`** (293行, 9929字节)
- **角色**: 实现具体的 `TensorTrain{ValueType, N}` 结构体
- **核心字段**:
  ```julia
  struct TensorTrain{ValueType, N}
      sitetensors::Vector{Array{ValueType, N}}  # N-维张量数组
  ```
- **核心功能**:
  - 张量列的构造与转换
  - `compress!` 方法：使用 LU/SVD 进行张量压缩
  - `fulltensor`: 将TT格式还原为全张量
  - `TensorTrainFit`: 用于拟合噪声数据的特殊结构

#### 3. **`tensorci1.jl`** (581行, 20671字节)
- **角色**: 实现 TCI1 算法（第一代交叉插值算法）
- **核心结构体**:
  ```julia
  mutable struct TensorCI1{ValueType} <: AbstractTensorTrain{ValueType}
      Iset::Vector{IndexSet{MultiIndex}}      # 左索引集合
      Jset::Vector{IndexSet{MultiIndex}}      # 右索引集合
      localdims::Vector{Int}                   # 局部维度
      T::Vector{Matrix{ValueType}}             # 张量核心
      P::Vector{Matrix{ValueType}}             # 投影矩阵
      aca::Vector{MatrixCI{ValueType}}         # 自适应交叉近似
      Pi::Vector{Matrix{ValueType}}            # 辅助矩阵
      PiIset, PiJset::Vector{IndexSet}        # 辅助索引集
      pivoterrors::Vector{Float64}             # 枢轴误差
      maxsamplevalue::Float64                  # 最大采样值
  ```
- **算法逻辑**: 单向扫描（half-sweep）策略

#### 4. **`tensorci2.jl`** (998行, 34875字节) - **最复杂的文件**
- **角色**: 实现 TCI2 算法（改进的双向扫描版本）
- **核心结构体**:
  ```julia
  mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}
      Iset::Vector{Vector{MultiIndex}}        # 左索引集（嵌套向量）
      Jset::Vector{Vector{MultiIndex}}        # 右索引集
      localdims::Vector{Int}                   # 局部维度
      sitetensors::Vector{Array{ValueType,3}} # 3D站点张量
      bonderrors::Vector{Float64}              # 键误差
      pivoterrors::Vector{Vector{Float64}}     # 枢轴误差（多层）
      maxsamplevalue::Float64                  # 最大采样值
      Iset_history, Jset_history::Vector{...} # 历史记录（用于嵌套性检查）
  ```
- **关键方法**:
  - `sweep0site!`, `sweep1site!`, `sweep2site!`: 多种扫描策略
  - `addglobalpivots!`: 全局枢轴添加
  - `makecanonical!`: 规范化操作

#### 5. **`indexset.jl`** (75行, 1913字节)
- **角色**: 定义 `IndexSet{T}` 结构，维护索引的双向映射
- **数据结构**:
  ```julia
  struct IndexSet{T}
      toint::Dict{T, Int}      # 索引 → 整数位置
      fromint::Vector{T}        # 整数位置 → 索引
  ```
- **作用**: 高效查找和去重操作，支持 O(1) 正反查询

#### 6. **`cachedtensortrain.jl`** (8235字节)
- **角色**: 带缓存的延迟求值张量列
- **用途**: 避免重复计算，优化函数采样效率

---

### 矩阵分解模块 (4个文件)

#### 7. **`matrixci.jl`** (287行, 8616字节)
- **角色**: 实现矩阵交叉插值的核心结构 `MatrixCI{T}`
- **数据结构**:
  ```julia
  mutable struct MatrixCI{T} <: AbstractMatrixCI{T}
      rowindices::Vector{Int}      # 枢轴行索引
      colindices::Vector{Int}       # 枢轴列索引
      pivotrows::Matrix{T}          # 枢轴行数据
      pivotcols::Matrix{T}          # 枢轴列数据
  ```
- **核心算法**: 
  - `AtimesBinv`: 计算 A*B^{-1} 的数值稳定方法（使用QR分解）
  - `addpivot!`: 动态添加枢轴
  - `crossinterpolate`: 矩阵交叉插值主循环

#### 8. **`matrixlu.jl`** (491行, 14414字节)
- **角色**: 实现 **Rank-Revealing LU (rrLU)** 分解
- **核心结构**:
  ```julia
  mutable struct rrLU{T}
      rowpermutation::Vector{Int}   # 行置换
      colpermutation::Vector{Int}   # 列置换
      L::Matrix{T}                  # 下三角因子
      U::Matrix{T}                  # 上三角因子
      leftorthogonal::Bool          # 正交性标志
      npivot::Int                   # 已选枢轴数
      error::Float64                # 分解误差
  ```
- **用途**: 低秩矩阵分解，支持 Rook 枢轴搜索策略

#### 9. **`matrixaca.jl`** (5362字节)
- **角色**: 自适应交叉近似（Adaptive Cross Approximation, ACA）
- **用途**: 在TCI1算法中用于增量式低秩更新

#### 10. **`matrixluci.jl`** (2418字节)
- **角色**: 结合LU分解的交叉插值变体
- **用途**: 提供更稳定的数值分解选项

---

### 辅助功能模块 (7个文件)

#### 11. **`cachedfunction.jl`** (205行, 8118字节)
- **角色**: 函数缓存包装器
- **核心结构**:
  ```julia
  struct CachedFunction{ValueType, K<:Union{UInt32,...,BigInt}}
      f::Function                # 原始函数
      localdims::Vector{Int}     # 维度信息
      cache::Dict{K, ValueType}  # 键值缓存
      coeffs::Vector{K}          # 哈希系数
  ```
- **优化**: 使用整数键（UInt128默认）进行快速哈希，避免重复函数求值

#### 12. **`batcheval.jl`** (4335字节)
- **角色**: 批量求值接口
- **作用**: 支持向量化函数调用，提升大规模采样性能

#### 13. **`util.jl`** (126行, 3246字节)
- **角色**: 通用工具函数集合
- **关键函数**:
  - `maxabs`: 更新最大绝对值
  - `optfirstpivot`: 优化首个枢轴位置（使用贪心扫描）
  - `randomsubset`: 随机子集采样

#### 14. **`contraction.jl`** (19091字节)
- **角色**: 实现张量网络收缩操作
- **用途**: 支持 `contract` 函数，压缩张量列

#### 15. **`conversion.jl`** (5854字节)
- **角色**: TCI对象与标准张量列格式的转换
- **用途**: 将 `TensorCI1/2` 转换为 `TensorTrain`

#### 16. **`integration.jl`** (1747字节)
- **角色**: 数值积分接口（使用 QuadGK）
- **用途**: 在连续域上进行插值积分

#### 17. **`globalsearch.jl`** (3803字节) & **`globalpivotfinder.jl`** (6037字节)
- **角色**: 全局枢轴搜索策略
- **用途**: TCI2算法中识别最佳全局枢轴位置

#### 18. **`abstractmatrixci.jl`** (3110字节)
- **角色**: 矩阵交叉插值的抽象接口
- **用途**: 为多种MatrixCI实现提供统一API

#### 19. **`sweepstrategies.jl`** (190字节)
- **角色**: 扫描策略枚举（可能是占位符或配置文件）

---

## 2. 依赖分析

### Julia 标准库依赖
1. **`LinearAlgebra`**: 所有矩阵/向量运算、QR/SVD分解、范数计算
2. **`Random`**: 随机采样、初始枢轴选择
3. **`Base`**: 运算符重载（`+`, `==`, `getindex`等）

### 第三方依赖（来自 `Project.toml`）
1. **`BitIntegers`** (v0.3.5)
   - **作用**: 提供 `UInt256` 等大整数类型
   - **C++映射**: 需要自定义大整数类或使用 GMP/Boost.Multiprecision

2. **`EllipsisNotation`** (v1)
   - **作用**: 支持 `A[.., j]` 语法（省略符号）
   - **C++映射**: 需手动展开或使用 Eigen 切片

3. **`QuadGK`** (v2.9)
   - **作用**: 自适应高斯积分
   - **C++映射**: 可用 GSL (GNU Scientific Library) 或 Boost.Math

### 可选扩展（Weak Dependencies）
- **`ITensors`/`ITensorMPS`**: 与张量网络社区生态集成
- **C++移植时可忽略**

---

## 3. C++ 移植所需库

| Julia 依赖 | C++ 替代方案 | 备注 |
|-----------|------------|------|
| `LinearAlgebra` | **Eigen** 或 **Armadillo** | 推荐 Eigen（header-only） |
| SVD/QR分解 | **LAPACK** + **BLAS** | Eigen内置，或直接调用 MKL |
| `BitIntegers` | **GMP** 或 **Boost.Multiprecision** | 用于超大键哈希 |
| `Dict` | `std::unordered_map` | 标准库 |
| `Vector{T}` | `std::vector<T>` | 标准库 |
| `QuadGK` | **GSL** 或 **Boost.Math** | 数值积分 |
| 动态类型 | 模板 `template<typename T>` | 编译时多态 |

---

## 4. 代码组织结构总结

```
src/
├── [入口] TensorCrossInterpolation.jl
│
├── [核心数据结构]
│   ├── abstracttensortrain.jl    (抽象基类)
│   ├── tensortrain.jl             (具体TT实现)
│   ├── tensorci1.jl               (TCI1算法)
│   ├── tensorci2.jl               (TCI2算法 ⭐最复杂)
│   ├── indexset.jl                (索引管理)
│   └── cachedtensortrain.jl       (缓存优化)
│
├── [矩阵分解后端]
│   ├── abstractmatrixci.jl        (抽象接口)
│   ├── matrixci.jl                (标准CI)
│   ├── matrixlu.jl                (rrLU分解 ⭐核心)
│   ├── matrixaca.jl               (ACA算法)
│   └── matrixluci.jl              (LU+CI混合)
│
├── [性能优化]
│   ├── cachedfunction.jl          (函数缓存)
│   └── batcheval.jl               (批量求值)
│
├── [算法辅助]
│   ├── globalsearch.jl            (全局搜索)
│   ├── globalpivotfinder.jl       (枢轴选择)
│   └── sweepstrategies.jl         (扫描策略)
│
└── [工具与接口]
    ├── util.jl                    (通用工具)
    ├── contraction.jl             (张量收缩)
    ├── conversion.jl              (格式转换)
    └── integration.jl             (数值积分)
```

---

## 5. 关键观察

### 代码复杂度分布
- **最复杂**: `tensorci2.jl` (998行) - 包含多种扫描策略和全局优化
- **核心算法**: `matrixlu.jl` (491行) - 数值线性代数核心
- **基础设施**: `abstracttensortrain.jl` - 定义整个类型系统

### Julia 特有特性使用情况
1. **参数化类型**: `{ValueType}`, `{T}`, `{V, N}` 广泛使用
   - C++对应: `template<typename ValueType, size_t N>`
2. **多重分派**: 同名函数有多个重载
   - C++对应: 函数重载 + SFINAE
3. **宏系统**: `@doc`, `@inbounds`, `@fastmath`
   - C++对应: 编译器提示 (`[[likely]]`, `#pragma`)
4. **广播操作**: `f.(x, y')` 
   - C++对应: Eigen 的 `.array()` 操作

---

## 结论

该代码库结构清晰，采用分层设计：
- **顶层**: TCI1/TCI2 算法实现
- **中层**: 矩阵分解与交叉插值
- **底层**: 线性代数与缓存优化

C++ 移植的主要挑战在于：
1. **动态类型系统** → 需要设计良好的模板架构
2. **内存管理** → Julia 的自动GC vs C++手动/智能指针
3. **矩阵库选择** → 建议使用 Eigen（语法接近Julia）
