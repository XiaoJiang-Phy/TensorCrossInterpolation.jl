# 文档 3: 算法逻辑流 (Algorithm Logic Flow)

## 概述
本文档基于实际代码（引用具体行号），还原两种交叉插值算法（TCI1 和 TCI2）的执行流程，并详细列出所有数学操作。

---

## 1. 用户入口函数

### TCI2 算法入口：`crossinterpolate2`

**文件**: `tensorci2.jl:943-953`

```julia
function crossinterpolate2(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))];
    kwargs...
) where {ValueType,N}
    tci = TensorCI2{ValueType}(f, localdims, initialpivots)
    ranks, errors = optimize!(tci, f; kwargs...)
    return tci, ranks, errors
end
```

**参数说明**:
- `ValueType`: 函数返回值类型（必须显式指定）
- `f`: 被插值函数 `f(v::Vector{Int}) -> ValueType`
- `localdims`: 每个维度的局部大小，例如 `[10, 10, 10, 10]` 表示4维函数，每维取值1-10
- `initialpivots`: 初始枢轴位置（默认全1向量）

**流程**:
1. 构造 `TensorCI2` 对象（初始化索引集）
2. 调用 `optimize!` 进行迭代优化
3. 返回优化后的 TCI 对象、秩历史、误差历史

---

## 2. TCI2 核心算法：`optimize!`

**文件**: `tensorci2.jl:700-850`

### 算法伪代码
```
输入: f (函数), localdims (维度), tolerance (精度), maxbonddim (最大键维度)
输出: 优化后的 TensorCI2 对象

1. 初始化：
   - tci = TensorCI2{ValueType}(f, localdims, initialpivots)
   - errors = [], ranks = [], nglobalpivots = []

2. FOR iter = 1 TO maxiter:
   a) 计算自适应容差: abstol = tolerance * maxsamplevalue
   
   b) 执行双站点扫描:
      sweep2site!(tci, f, ...) → 更新索引集 Iset, Jset
   
   c) 搜索全局枢轴:
      globalpivots = globalpivotfinder(tci, f, abstol)
      addglobalpivots!(tci, globalpivots)
   
   d) 记录历史:
      push!(errors, pivoterror(tci))
      push!(ranks, rank(tci))
      push!(nglobalpivots, length(globalpivots))
   
   e) 检查收敛:
      IF convergencecriterion(...):
         BREAK

3. 清理优化（单站点扫描移除冗余枢轴）:
   sweep1site!(tci, f, ...)

4. 返回 ranks, errors
```

**代码证据**:
- **行768-831**: 主循环实现
- **行777-787**: 双站点扫描调用
- **行803-810**: 全局枢轴搜索
- **行823-830**: 收敛判定

---

## 3. 双站点扫描：`sweep2site!`

**文件**: `tensorci2.jl:855-916`

### 执行流程

```
FOR each iteration:
  1. 记录历史:
     Iset_history.push(deepcopy(Iset))
     Jset_history.push(deepcopy(Jset))
  
  2. 选择扫描方向 (forward 或 backward):
     
     IF forward:
       FOR bondindex = 1 TO n-1:
         updatepivots!(tci, bondindex, f, leftorthogonal=true, ...)
     ELSE (backward):
       FOR bondindex = n-1 DOWNTO 1:
         updatepivots!(tci, bondindex, f, leftorthogonal=false, ...)
  
  3. 填充站点张量:
     fillsitetensors!(tci, f)
```

**关键特性**:
- **嵌套性保护** (`strictlynested=false` 时): 使用历史索引集作为 `extraIset/extraJset`，防止索引集收缩
- **行883-895**: 前向扫描实现
- **行896-909**: 后向扫描实现

---

## 4. 枢轴更新：`updatepivots!`

**文件**: `tensorci2.jl:510-607`

### 核心逻辑（最关键的函数）

```
输入: tci, bond_index (b), f, leftorthogonal, abstol, maxbonddim

1. 构造组合索引集:
   Icombined = union(kronecker(Iset[b], localdims[b]), extraIset)
   Jcombined = union(kronecker(localdims[b+1], Jset[b+1]), extraJset)

2. 采样枢轴矩阵 Pi:
   IF pivotsearch == :full:
      Pi = filltensor(f, Icombined, Jcombined, Val(0))  # 全采样
      luci = MatrixLUCI(Pi, reltol, abstol, maxbonddim, leftorthogonal)
   
   ELSE IF pivotsearch == :rook:
      # 只采样必要的行/列（Rook 策略）
      I0 = 初始行索引（来自上一次的 Iset[b+1]）
      J0 = 初始列索引（来自上一次的 Jset[b]）
      luci = MatrixLUCI(ValueType, SubMatrix{f, Icombined, Jcombined}, 
                        (m, n), I0, J0, ..., pivotsearch=:rook)

3. 更新索引集:
   tci.Iset[b+1] = Icombined[rowindices(luci)]
   tci.Jset[b] = Jcombined[colindices(luci)]

4. 更新站点张量:
   setsitetensor!(tci, b, left(luci))
   setsitetensor!(tci, b+1, right(luci))

5. 更新误差:
   updateerrors!(tci, b, pivoterrors(luci))
```

**代码位置**:
- **行526-527**: 构造组合索引集
- **行529-551**: 全搜索模式（**行531-535**: 批量采样矩阵）
- **行552-595**: Rook 搜索模式（稀疏采样）
- **行599-606**: 更新结果

**数学操作**:
- `filltensor`: 批量函数求值（见下文§7）
- `MatrixLUCI`: 秩揭示LU分解（见下文§5）
- `kronecker`: 索引集的笛卡尔积

---

## 5. 矩阵交叉插值：`MatrixLUCI`

**文件**: `matrixluci.jl`（该文件调用 `matrixlu.jl` 中的 `rrlu`）

### 秩揭示 LU 分解流程

**定义**: `matrixlu.jl:193-207`

```
输入: 矩阵 A (m × n), reltol, abstol, maxrank

1. 初始化:
   lu = rrLU{T}(m, n)
   lu.rowpermutation = [1, 2, ..., m]
   lu.colpermutation = [1, 2, ..., n]
   lu.npivot = 0

2. WHILE lu.npivot < maxrank:
   a) 在 A[k:end, k:end] 中找最大元素:
      (row, col) = submatrixargmax(abs, A, k:m, k:n)
      lu.error = abs(A[row, col])
   
   b) 误差检查:
      threshold = max(abstol, reltol * max_abs(A))
      IF lu.error < threshold:
         BREAK
   
   c) 行列置换:
      swap_row!(A, k, row)
      swap_col!(A, k, col)
   
   d) 消元（Schur补更新）:
      pivot = A[k, k]
      A[k+1:end, k+1:end] -= A[k+1:end, k] * A[k, k+1:end] / pivot
   
   e) npivot += 1

3. 提取 L, U 因子
4. 返回 rrLU 对象
```

**代码证据**:
- **行146-198**: `_optimizerrlu!` 主循环
- **行129-141**: 枢轴选择逻辑（`submatrixargmax`）
- **行333-335**: Schur补更新实现

**数学操作**:
- **QR 分解**: `AtimesBinv(A, B)` 使用 QR 避免直接求逆（`matrixci.jl:2-16`）
- **LU 分解**: 带行列主元的 Gaussian 消元

---

## 6. 全局枢轴搜索（Global Pivot Search）

**文件**: `globalpivotfinder.jl` + `globalsearch.jl`

### 搜索策略

**主函数**: `tensorci2.jl:803-810`

```julia
input = GlobalPivotSearchInput(tci)  # 构造搜索输入
globalpivots = finder(input, f, abstol; verbosity, rng)
addglobalpivots!(tci, globalpivots)
```

### `DefaultGlobalPivotFinder` 工作流程

**伪代码**（基于 `globalpivotfinder.jl` 推断）:
```
输入: tci (当前插值), f (原函数), abstol (容差), nsearch (搜索次数)

1. 转换 TCI 为 TensorTrain:
   tt = TensorTrain(tci)

2. FOR _ = 1 TO nsearch:
   a) 随机采样候选点:
      candidate = random_multi_index(localdims)
   
   b) 计算插值误差:
      approx = evaluate(tt, candidate)
      exact = f(candidate)
      error = abs(approx - exact)
   
   c) IF error > abstol * tolmarginglobalsearch:
         候选池.add((error, candidate))

3. 按误差排序，选择前 maxnglobalpivot 个
4. 返回选定的全局枢轴
```

**代码位置**:
- **行804**: 构造 `GlobalPivotSearchInput`（包含当前 TT、索引集）
- **行805-809**: 调用查找器
- **行810**: 添加找到的枢轴到索引集

---

## 7. 批量函数求值：`filltensor`

**文件**: `tensorci2.jl:290-312`

### 功能
将多索引函数 `f` 批量求值，填充多维张量。

### 代码分析
```julia
function filltensor(
    ::Type{ValueType},
    f,
    localdims::Vector{Int},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex},
    ::Val{M}
)::Array{ValueType,M+2} where {ValueType,M}
    nl = length(first(Iset))      # 左索引长度
    nr = length(first(Jset))      # 右索引长度
    ncent = N - nl - nr           # 中心索引数量
    
    # 调用批量求值分发器
    result = _batchevaluate_dispatch(ValueType, f, localdims, Iset, Jset, Val(ncent))
    
    # 重塑为正确形状: (|Iset|, localdim[nl+1], ..., localdim[nl+ncent], |Jset|)
    return reshape(result, (length(Iset), localdims[nl+1:nl+ncent]..., length(Jset)))
end
```

**批量求值策略**（来自 `batcheval.jl`）:
```
FOR each (i, leftindex) in enumerate(Iset):
  FOR each (j, rightindex) in enumerate(Jset):
    FOR each k in product(localdims[nl+1:nl+ncent]):
      fullindex = [leftindex..., k..., rightindex...]
      result[i, k..., j] = f(fullindex)
```

**性能优化**:
- 如果 `f` 是 `BatchEvaluator` 类型，调用其批量接口避免循环开销
- 使用 `CachedFunction` 包装可避免重复计算（见 `cachedfunction.jl:90-170`）

---

## 8. 站点张量计算：`setsitetensor!`

**文件**: `tensorci2.jl:367-394`

### 左正交模式（`leftorthogonal=true`）

```
1. 扩展左索引集:
   Is = kronecker(Iset[b], localdims[b])  # 长度: |Iset[b]| × localdims[b]

2. 采样 Pi1 矩阵:
   Pi1 = filltensor(f, Iset[b], Jset[b], Val(1))  # 形状: (|Is|, |Jset[b]|)

3. 采样枢轴矩阵 P:
   P = filltensor(f, Iset[b+1], Jset[b], Val(0))  # 形状: (|Iset[b+1]|, |Jset[b]|)

4. 求解张量核心 T:
   Tmat = transpose(P^T \ Pi1^T)  # 等价于 Pi1 * inv(P)
   T[b] = reshape(Tmat, (|Iset[b]|, localdims[b], |Iset[b+1]|))
```

**数学关系**:
```
Pi1[i, j] = f([Iset[b][i], localdims[b], Jset[b][j]])
P[i', j] = f([Iset[b+1][i'], Jset[b][j]])

站点张量: T[i, s, i'] 使得:
  Pi1 ≈ T[:, :, :] 在索引上求和后等于 T * P
```

**代码位置**:
- **行372-376**: 索引扩展与 Pi1 采样
- **行385-387**: 枢轴矩阵 P 采样
- **行391-392**: 线性求解（使用 `\` 运算符，内部调用 LAPACK）

---

## 9. 单站点扫描：`sweep1site!`

**文件**: `tensorci2.jl:402-461`

### 与双站点扫描的区别

| 特性 | 单站点 (`sweep1site!`) | 双站点 (`sweep2site!`) |
|-----|----------------------|---------------------|
| 更新范围 | 只更新一个索引集 | 同时更新相邻两个索引集 |
| 枢轴矩阵维度 | `(|Iset ⊗ localdim|, |Jset|)` | `(|Iset ⊗ localdim[b]|, |localdim[b+1] ⊗ Jset|)` |
| 用途 | 清理冗余枢轴、最终优化 | 主要插值改进 |
| 调用位置 | `optimize!` 结束前 | 每次迭代的核心 |

### 前向扫描流程
```
FOR b = 1 TO n-1:
  1. 扩展左索引:
     Is = kronecker(Iset[b], localdims[b])
  
  2. 采样 Pi 矩阵:
     Pi = filltensor(f, Iset[b], Jset[b], Val(1))
  
  3. LU 分解:
     luci = MatrixLUCI(Pi, reltol, abstol, maxbonddim, leftorthogonal=true)
  
  4. 更新索引集:
     Iset[b+1] = Is[rowindices(luci)]  # 只更新右侧索引集
  
  5. 更新站点张量:
     sitetensors[b] = left(luci)
```

**代码位置**:
- **行420-446**: 主循环（支持前向/后向）
- **行434-435**: 索引集更新
- **行436-438**: 站点张量更新

---

## 10. 收敛判定：`convergencecriterion`

**文件**: `tensorci2.jl:609-628`

### 判定条件（所有条件必须同时满足）

```julia
function convergencecriterion(
    ranks::Vector{Int},
    errors::Vector{Float64},
    nglobalpivots::Vector{Int},
    tolerance::Float64,
    maxbonddim::Int,
    ncheckhistory::Int
)::Bool
    # 条件1: 最近 ncheckhistory 次的误差都小于容差
    all(last(errors, ncheckhistory) .< tolerance) &&
    
    # 条件2: 最近一次没有新全局枢轴
    all(last(nglobalpivots, ncheckhistory) .== 0) &&
    
    # 条件3: 秩已经稳定（不再增长）
    minimum(last(ranks, ncheckhistory)) == last(ranks)
    
    # 或者 条件4: 秩已经达到上限
    || all(last(ranks, ncheckhistory) .>= maxbonddim)
end
```

**逻辑**:
- **提前停止**: 插值误差足够小 + 无新枢轴 + 秩稳定
- **强制停止**: 秩达到用户设定的最大键维度

---

## 11. 数学操作汇总

### 线性代数操作

| 操作 | Julia 代码 | 调用库 | 文件位置 |
|-----|-----------|-------|---------|
| **QR 分解** | `qr(B')` | `LinearAlgebra.qr` | `matrixci.jl:6-16` |
| **矩阵求逆求解** | `B \ A'` | LAPACK (`gesv`) | `tensorci2.jl:391` |
| **奇异值分解 (SVD)** | `svd(A)` | LAPACK (`gesvd`) | `tensortrain.jl:110-132` |
| **LU 分解** | `_optimizerrlu!(...)` | 自定义实现 | `matrixlu.jl:146-198` |
| **矩阵乘法** | `A * B` | BLAS (`gemm`) | 广泛使用 |
| **向量范数** | `norm(v)` | BLAS (`nrm2`) | `abstracttensortrain.jl:291` |
| **最大绝对值** | `maximum(abs, A)` | Julia 标准库 | `matrixlu.jl:153` |

### 张量操作

| 操作 | 实现方式 | 复杂度 |
|-----|---------|-------|
| **张量收缩** | `prod(T[:, i, :] for T in tt)` | O(n × r²) |
| **张量求和** | `sum(T, dims=(1,2))` | O(n × d × r²) |
| **张量重塑** | `reshape(A, dims...)` | O(1) (视图操作) |
| **Kronecker 积** | `[is..., j] for is in I, j in 1:d]` | O(\|I\| × d) |

---

## 12. 完整算法流程图（TCI2）

```
用户调用:
  crossinterpolate2(Float64, f, [10, 10, 10, 10])
    ↓
┌─────────────────────────────────────────┐
│ 1. 初始化 TensorCI2                      │
│    - Iset, Jset = 初始枢轴索引           │
│    - maxsamplevalue = |f(initialpivot)|  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. optimize!(tci, f)                    │
│    主迭代循环 (最多 maxiter=20 次)        │
└─────────────────────────────────────────┘
    ↓
    ┌───────────→ [iter = 1, 2, ..., maxiter]
    │
    ├─→ 2.1 计算容差: abstol = tolerance * maxsamplevalue
    │
    ├─→ 2.2 双站点扫描: sweep2site!(tci, f, ...)
    │       ↓
    │       FOR each bond b = 1 TO n-1:
    │         ├─→ updatepivots!(tci, b, f, ...)
    │         │     ├─→ 构造 Icombined, Jcombined
    │         │     ├─→ 采样 Pi = filltensor(...)
    │         │     ├─→ LU分解 luci = MatrixLUCI(Pi, ...)
    │         │     ├─→ 更新 Iset[b+1], Jset[b]
    │         │     └─→ 更新 sitetensors[b]
    │         └─→ [下一个键]
    │
    ├─→ 2.3 搜索全局枢轴: globalpivots = finder(...)
    │       ├─→ 随机采样候选点
    │       ├─→ 计算 |f(x) - tci(x)|
    │       └─→ 返回误差最大的 maxnglobalpivot 个点
    │
    ├─→ 2.4 添加枢轴: addglobalpivots!(tci, globalpivots)
    │
    ├─→ 2.5 记录: push!(errors, pivoterror(tci))
    │            push!(ranks, rank(tci))
    │
    ├─→ 2.6 检查收敛: IF convergencecriterion(...):
    │                   BREAK
    └───────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. 清理: sweep1site!(tci, f)            │
│    - 移除冗余枢轴                        │
│    - 确保站点张量一致性                   │
└─────────────────────────────────────────┘
    ↓
返回: (tci, ranks, errors)
```

---

## 13. TCI1 vs TCI2 关键差异

| 特性 | TCI1 (`tensorci1.jl`) | TCI2 (`tensorci2.jl`) |
|-----|----------------------|----------------------|
| **扫描策略** | 单向（half-sweep） | 双向（full-sweep + 2site） |
| **索引集类型** | `IndexSet{MultiIndex}` | `Vector{MultiIndex}` |
| **枢轴更新** | `addpivot!` 逐个添加 | `updatepivots!` 批量更新 |
| **站点张量** | 2D 矩阵 `T[p]::Matrix` | 3D 数组 `T[p]::Array{_, 3}` |
| **全局枢轴** | `addglobalpivot!` (单点) | `addglobalpivots!` (批量) |
| **误差跟踪** | `pivoterrors::Vector{Float64}` | `pivoterrors::Vector{Vector{Float64}}` |
| **收敛速度** | 较慢 | **更快**（推荐） |
| **代码复杂度** | 581 行 | **998 行** |

**推荐使用**: TCI2（更稳定、更快）

---

## 14. 关键数值稳定性技巧

### 1. 避免矩阵求逆：使用 QR 分解
```julia
# matrixci.jl:2-16
function AtimesBinv(A::Matrix, B::Matrix)
    qr_decomp = qr(transpose(B))
    return transpose(qr_decomp.Q * (qr_decomp.R \ transpose(A)))
end
```
**原理**: 计算 `A * B^{-1}` 时，通过 QR 分解避免显式求逆，减少条件数放大。

### 2. 自适应容差归一化
```julia
# tensorci2.jl:769-770
errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
abstol = tolerance * errornormalization
```
**作用**: 当函数幅值未知时，动态调整容差避免over-/under-fitting。

### 3. 秩截断与误差控制
```julia
# matrixlu.jl:153-159
threshold = max(abstol, reltol * maximum(abs, A))
if lu.error < threshold:
    break
```
**策略**: 结合绝对容差（`abstol`）和相对容差（`reltol`），适应不同量级的函数。

---

## 15. 性能瓶颈识别

### 热点 1: `filltensor` (批量采样)
- **位置**: `tensorci2.jl:290-312`
- **复杂度**: O(|Iset| × |Jset| × d × T_f)，其中 T_f 是单次函数求值时间
- **优化**: 使用 `CachedFunction` 或 `BatchEvaluator`

### 热点 2: `MatrixLUCI` (LU 分解)
- **位置**: `matrixluci.jl` → `matrixlu.jl:146-198`
- **复杂度**: O(r² × min(m, n))，其中 r 是秩
- **优化**: Rook 搜索 (`:rook`) 避免全矩阵采样

### 热点 3: 全局枢轴搜索
- **位置**: `globalpivotfinder.jl`
- **复杂度**: O(nsearch × T_evaluate)
- **优化**: 减少 `nsearchglobalpivot` 参数

---

## 总结

### 核心算法链条
```
用户函数 f
  ↓
批量采样 → filltensor
  ↓
秩揭示分解 → MatrixLUCI → rrLU
  ↓
索引集更新 → Iset, Jset
  ↓
张量核心 → sitetensors
  ↓
误差评估 → pivoterrors
  ↓
全局优化 → globalpivotfinder
  ↓
收敛检查 → convergencecriterion
```

### 数学操作密集度
- **高频**: 矩阵乘法 (BLAS Level 3)、QR 分解、线性求解
- **中频**: SVD（仅在压缩时）、范数计算
- **低频**: 全局搜索（随机采样）

### C++ 移植建议
1. **使用 Eigen** 处理所有线性代数（性能接近 BLAS）
2. **重点优化** `rrLU` 实现（考虑调用 LAPACK 的 `dgetrf`）
3. **批量求值接口** 必须支持向量化以匹配性能
