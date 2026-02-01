# ===================================================================
# matrixaca.jl - 自适应交叉近似 (Adaptive Cross Approximation)
# ===================================================================
# 这个文件实现了MatrixACA类型，它是自适应交叉近似(ACA)算法的实现。
#
# ACA是另一种低秩矩阵近似方法，与标准交叉插值略有不同：
#   A ≈ ∑ₖ αₖ * uₖ * vₖᵀ
# 其中 uₖ 和 vₖ 是列向量和行向量，αₖ 是标量权重。
#
# ACA的优点是可以增量更新：每次添加新枢轴只需要更新 uₖ 和 vₖ，
# 不需要重新计算之前的分量。
#
# 参考文献: Kumar 2016
# ===================================================================

"""
    mutable struct MatrixACA{T} <: AbstractMatrixCI{T}

自适应交叉近似(Adaptive Cross Approximation)的数据结构。

# 数学描述
给定一个 m×n 的矩阵 A，ACA构建如下分解：
```math
A ≈ ∑_{k=1}^{r} α_k u_k v_k^T
```
其中：
- `uₖ` 是 m×1 的列向量
- `vₖ` 是 1×n 的行向量
- `αₖ` 是标量权重，通常取 1/δₖ，δₖ 是枢轴值

# 类型参数
- `T`: 矩阵元素类型

# 字段
- `rowindices::Vector{Int}`: 选中的行索引（枢轴的x坐标）
- `colindices::Vector{Int}`: 选中的列索引（枢轴的y坐标）
- `u::Matrix{T}`: u向量的集合，形状 (nrows, npivots)
  - `u[:, k]` 是第k个u向量
- `v::Matrix{T}`: v向量的集合，形状 (npivots, ncols)
  - `v[k, :]` 是第k个v向量
- `alpha::Vector{T}`: 权重向量
  - `alpha[k] = 1 / A[xₖ, yₖ]`

# 重构公式
```math
Ã[i, j] = ∑_k u[i, k] * α[k] * v[k, j]
```

# 与MatrixCI的区别
- MatrixCI存储原始的枢轴行和列
- MatrixACA存储修正后的u和v向量，使得更新更高效

# 示例
```julia
A = rand(100, 50)
aca = MatrixACA(A, argmax(abs.(A)))  # 从最大元素开始
for i in 1:10
    addpivot!(aca, A)  # 添加更多枢轴
end
approx = Matrix(aca)  # 重构近似矩阵
```
"""
mutable struct MatrixACA{T} <: AbstractMatrixCI{T}
    rowindices::Vector{Int}   # 枢轴行索引
    colindices::Vector{Int}   # 枢轴列索引

    u::Matrix{T}   # uₖ(x) 向量：形状 (nrows, npivots)
    v::Matrix{T}   # vₖ(y) 向量：形状 (npivots, ncols)
    alpha::Vector{T}  # α = 1/δ：权重向量，长度 npivots

    """
        MatrixACA(::Type{T}, nrows::Int, ncols::Int) where {T<:Number}
    
    创建一个空的MatrixACA（没有枢轴）。
    """
    function MatrixACA(
        ::Type{T},
        nrows::Int, ncols::Int
    ) where {T<:Number}
        # zeros(nrows, 0) 创建 nrows×0 的空矩阵
        return new{T}(Int[], Int[], zeros(nrows, 0), zeros(0, ncols), T[])
    end

    """
        MatrixACA(A::AbstractMatrix{T}, firstpivot) where {T<:Number}
    
    从矩阵和第一个枢轴创建MatrixACA。
    
    # 参数
    - `A`: 原始矩阵
    - `firstpivot`: 第一个枢轴位置 (i, j)
    """
    function MatrixACA(
        A::AbstractMatrix{T},
        firstpivot::Union{CartesianIndex{2},Tuple{Int,Int},Pair{Int,Int}}
    ) where {T<:Number}
        return new{T}(
            [firstpivot[1]], [firstpivot[2]],  # 行列索引
            A[:, [firstpivot[2]]],              # 第一个u向量 = 第一个枢轴列
            A[[firstpivot[1]], :],              # 第一个v向量 = 第一个枢轴行
            [1 / A[firstpivot[1], firstpivot[2]]]  # 第一个alpha = 1/枢轴值
        )
    end
end

# ===================================================================
# 访问器函数
# ===================================================================

"""
    nrows(aca::MatrixACA)

获取原矩阵的行数。
"""
function nrows(aca::MatrixACA)
    return size(aca.u, 1)
end

"""
    ncols(aca::MatrixACA)

获取原矩阵的列数。
"""
function ncols(aca::MatrixACA)
    return size(aca.v, 2)
end

"""
    npivots(aca::MatrixACA)

获取当前的枢轴数量。
"""
function npivots(aca::MatrixACA)
    return size(aca.u, 2)
end

"""
    rank(ci::MatrixACA{T}) where {T}

获取近似的秩（等于枢轴数量）。
"""
function rank(ci::MatrixACA{T}) where {T}
    return length(ci.rowindices)
end

"""
    Base.isempty(ci::MatrixACA{T}) where {T}

检查是否没有枢轴。
"""
function Base.isempty(ci::MatrixACA{T}) where {T}
    return Base.isempty(ci.colindices)
end

"""
    availablerows(aca::MatrixACA{T}) where {T}

获取尚未被选为枢轴的行索引。
"""
function availablerows(aca::MatrixACA{T}) where {T}
    return setdiff(1:nrows(aca), aca.rowindices)
end

"""
    availablecols(aca::MatrixACA{T}) where {T}

获取尚未被选为枢轴的列索引。
"""
function availablecols(aca::MatrixACA{T}) where {T}
    return setdiff(1:ncols(aca), aca.colindices)
end

# ===================================================================
# ACA核心算法：计算u和v向量
# ===================================================================

"""
    uk(aca::MatrixACA{T}, A) where {T}

计算第k个u向量（内部函数）。

# 数学公式
uₖ(x) = A(x, yₖ) - ∑_{l=1}^{k-1} [vₗ(yₖ) / uₗ(xₗ)] * uₗ(x)

# 说明
uₖ是原矩阵第yₖ列减去之前分量在yₖ处的贡献。
这确保了ACA分量互不干扰。

# 参数
- `aca`: MatrixACA对象
- `A`: 原始矩阵

# 返回值
- 长度为nrows的向量
"""
function uk(aca::MatrixACA{T}, A) where {T}
    k = length(aca.colindices)
    yk = aca.colindices[end]  # 最新的列索引
    
    # 从原矩阵的列开始
    result = copy(A[:, yk])
    
    u, v = aca.u, aca.v
    
    # 减去之前分量的贡献
    for l in 1:k-1
        xl = aca.rowindices[l]
        # v[l, yk] / u[xl, l] 是归一化因子
        result -= (v[l, yk] / u[xl, l]) * u[:, l]
    end
    
    return result
end

"""
    addpivotcol!(aca::MatrixACA{T}, a::AbstractMatrix{T}, yk::Int) where {T}

添加一个新的枢轴列。

# 参数
- `aca`: MatrixACA对象（会被修改）
- `a`: 原始矩阵
- `yk`: 要添加的列索引
"""
function addpivotcol!(aca::MatrixACA{T}, a::AbstractMatrix{T}, yk::Int) where {T}
    push!(aca.colindices, yk)
    # 计算并存储新的u向量
    aca.u = hcat(aca.u, uk(aca, a))
end

"""
    vk(aca::MatrixACA{T}, A) where {T}

计算第k个v向量（内部函数）。

# 数学公式
vₖ(y) = A(xₖ, y) - ∑_{l=1}^{k-1} [uₗ(xₖ) / uₗ(xₗ)] * vₗ(y)

# 说明
vₖ是原矩阵第xₖ行减去之前分量在xₖ处的贡献。

# 参数
- `aca`: MatrixACA对象
- `A`: 原始矩阵

# 返回值
- 长度为ncols的向量
"""
function vk(aca::MatrixACA{T}, A) where {T}
    k = length(aca.rowindices)
    xk = aca.rowindices[end]  # 最新的行索引
    
    # 从原矩阵的行开始
    result = copy(A[xk, :])
    
    u, v = aca.u, aca.v
    
    # 减去之前分量的贡献
    for l in 1:k-1
        xl = aca.rowindices[l]
        result -= (u[xk, l] / u[xl, l]) * v[l, :]
    end
    
    return result
end

"""
    addpivotrow!(aca::MatrixACA{T}, a::AbstractMatrix{T}, xk::Int) where {T}

添加一个新的枢轴行。

# 参数
- `aca`: MatrixACA对象（会被修改）
- `a`: 原始矩阵
- `xk`: 要添加的行索引
"""
function addpivotrow!(aca::MatrixACA{T}, a::AbstractMatrix{T}, xk::Int) where {T}
    push!(aca.rowindices, xk)
    # 计算并存储新的v向量（转置为行向量）
    aca.v = vcat(aca.v, transpose(vk(aca, a)))
    # 计算并存储新的alpha权重
    push!(aca.alpha, 1 / aca.u[xk, end])
end

# ===================================================================
# 枢轴添加函数
# ===================================================================

"""
    addpivot!(aca::MatrixACA{T}, a::AbstractMatrix{T}, pivotindices) where {T}

在指定位置添加一个新枢轴。

# 参数
- `aca`: MatrixACA对象
- `a`: 原始矩阵
- `pivotindices`: 枢轴位置 (i, j)

# 算法（Kumar 2016）
1. 先添加列：计算uₖ
2. 再添加行：计算vₖ和αₖ
"""
function addpivot!(
    aca::MatrixACA{T},
    a::AbstractMatrix{T},
    pivotindices::Union{CartesianIndex{2},Tuple{Int,Int},Pair{Int,Int}}
) where {T}
    addpivotcol!(aca, a, pivotindices[2])
    addpivotrow!(aca, a, pivotindices[1])
end

"""
    addpivot!(aca::MatrixACA{T}, a::AbstractMatrix{T}) where {T}

自动选择并添加新枢轴。

# 算法（贪婪ACA）
1. 从最后一个v向量中选择绝对值最大的位置作为新列
2. 计算该列的u向量
3. 从新u向量中选择绝对值最大的位置作为新行
4. 计算该行的v向量和alpha权重
"""
function addpivot!(aca::MatrixACA{T}, a::AbstractMatrix{T}) where {T}
    # 步骤1: 选择新列
    availcols = availablecols(aca)
    # 在最后一个v向量中找最大元素（限制在可用列中）
    yk = availcols[argmax(abs.(aca.v[end, availcols]))]
    addpivotcol!(aca, a, yk)

    # 步骤2: 选择新行
    availrows = availablerows(aca)
    # 在最新u向量中找最大元素（限制在可用行中）
    xk = availrows[argmax(abs.(aca.u[availrows, end]))]
    addpivotrow!(aca, a, xk)
end

# ===================================================================
# 求值函数
# ===================================================================

"""
    submatrix(aca::MatrixACA{T}, rows, cols) where {T}

提取近似矩阵的子矩阵。

# 计算公式
Ã[rows, cols] = u[rows, :] * diag(α) * v[:, cols]

# 参数
- `rows`: 行选择器
- `cols`: 列选择器

# 返回值
- 近似矩阵的子矩阵
"""
function submatrix(
    aca::MatrixACA{T},
    rows::Union{AbstractVector{Int},Colon},
    cols::Union{AbstractVector{Int},Colon}
)::Matrix{T} where {T}
    if isempty(aca)
        return zeros(
            T,
            _lengthordefault(rows, nrows(aca)),
            _lengthordefault(cols, ncols(aca)))
    else
        r = rank(aca)
        # [CartesianIndex()] 用于添加一个新轴，实现广播
        newaxis = [CartesianIndex()]
        # u[rows, 1:r] 是 (nrows_selected, r)
        # alpha[1:r, newaxis] 是 (r, 1)
        # v[1:r, cols] 是 (r, ncols_selected)
        # 结果是 (nrows_selected, ncols_selected)
        return aca.u[rows, 1:r] * (aca.alpha[1:r, newaxis] .* aca.v[1:r, cols])
    end
end

"""
    Base.Matrix(aca::MatrixACA{T}) where {T}

将MatrixACA转换为普通矩阵。
"""
function Base.Matrix(aca::MatrixACA{T})::Matrix{T} where {T}
    return submatrix(aca, :, :)
end

"""
    evaluate(aca::MatrixACA{T}) where {T}

计算完整的近似矩阵。
"""
function evaluate(aca::MatrixACA{T})::Matrix{T} where {T}
    return submatrix(aca, :, :)
end

"""
    evaluate(aca::MatrixACA{T}, i::Int, j::Int) where {T}

计算位置(i, j)处的近似值。

# 计算公式
Ã[i, j] = ∑ₖ u[i, k] * α[k] * v[k, j]
"""
function evaluate(aca::MatrixACA{T}, i::Int, j::Int)::T where {T}
    return sum(aca.u[i, :] .* aca.alpha .* aca.v[:, j])
end

# ===================================================================
# 增量更新函数
# ===================================================================

"""
    setcols!(aca::MatrixACA{T}, newpivotrows::AbstractMatrix{T}, permutation::Vector{Int}) where {T}

在列排列变化后更新ACA对象。

# 说明
当枢轴列的顺序改变时（例如添加新的列索引），
需要更新v矩阵和列索引。

# 参数
- `aca`: MatrixACA对象（会被修改）
- `newpivotrows`: 新的枢轴行矩阵
- `permutation`: 旧列索引到新列索引的映射
"""
function setcols!(
    aca::MatrixACA{T},
    newpivotrows::AbstractMatrix{T},
    permutation::Vector{Int}
) where {T}
    # 更新列索引
    aca.colindices = permutation[aca.colindices]

    # 重排旧元素
    tempv = Matrix{T}(undef, size(newpivotrows))
    tempv[:, permutation] = aca.v
    aca.v = tempv

    # 插入新元素
    newindices = setdiff(1:size(newpivotrows, 2), permutation)
    for k in 1:size(newpivotrows, 1)
        aca.v[k, newindices] = newpivotrows[k, newindices]
        # 减去之前分量的贡献
        for l in 1:k-1
            aca.v[k, newindices] -= aca.v[l, newindices] *
                                    (aca.u[aca.rowindices[k], l] * aca.alpha[l])
        end
    end
end

"""
    setrows!(aca::MatrixACA{T}, newpivotcols::AbstractMatrix{T}, permutation::Vector{Int}) where {T}

在行排列变化后更新ACA对象。

# 说明
类似于setcols!，但用于行的更新。

# 参数
- `aca`: MatrixACA对象（会被修改）
- `newpivotcols`: 新的枢轴列矩阵
- `permutation`: 旧行索引到新行索引的映射
"""
function setrows!(
    aca::MatrixACA{T},
    newpivotcols::AbstractMatrix{T},
    permutation::Vector{Int}
) where {T}
    # 更新行索引
    aca.rowindices = permutation[aca.rowindices]

    # 重排旧元素
    tempu = Matrix{T}(undef, size(newpivotcols))
    tempu[permutation, :] = aca.u
    aca.u = tempu

    # 插入新元素
    newindices = setdiff(1:size(newpivotcols, 1), permutation)
    for k in 1:size(newpivotcols, 2)
        aca.u[newindices, k] = newpivotcols[newindices, k]
        # 减去之前分量的贡献
        for l in 1:k-1
            aca.u[newindices, k] -= aca.u[newindices, l] *
                                    (aca.v[l, aca.colindices[k]] * aca.alpha[l])
        end
    end
end
