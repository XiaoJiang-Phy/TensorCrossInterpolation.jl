# ===================================================================
# abstractmatrixci.jl - 矩阵交叉插值的抽象基类型
# ===================================================================
# 这个文件定义了所有矩阵交叉插值类型的公共接口。
#
# 矩阵交叉插值(Matrix Cross Interpolation)是一种低秩矩阵近似方法：
#   给定一个 m×n 矩阵 A，选择 r 个行索引 I 和 r 个列索引 J，
#   使得 A ≈ A[:, J] * (A[I, J])^(-1) * A[I, :]
#
# 这个分解只需要存储：
#   - r 列 (m×r 矩阵)
#   - r 行 (r×n 矩阵)
#   - r×r 的枢轴矩阵
# 共 (m+n+r)×r 个元素，当 r << min(m,n) 时远小于原始的 m×n
#
# AbstractMatrixCI是所有矩阵交叉插值类型的抽象父类型，
# 定义了它们共享的接口。
# ===================================================================

"""
    abstract type AbstractMatrixCI{T} end

矩阵交叉插值的抽象类型。

# 类型参数
- `T`: 矩阵元素的类型（如 Float64, ComplexF64）

# 子类型必须实现的方法
- `size(ci)`: 返回原始矩阵的尺寸
- `npivots(ci)`: 返回枢轴数量
- `rowindices(ci)`: 返回选中的行索引
- `colindices(ci)`: 返回选中的列索引
- `submatrix(ci, rows, cols)`: 获取近似矩阵的子矩阵
- `evaluate(ci, i, j)`: 计算位置(i,j)处的近似值

# 子类型
- `MatrixCI`: 标准矩阵交叉插值
- `MatrixACA`: 自适应交叉近似
- `MatrixLUCI`: 基于LU分解的交叉插值
"""
abstract type AbstractMatrixCI{T} end

"""
    Base.size(ci::AbstractMatrixCI)

返回交叉插值所表示的矩阵的尺寸。

# 返回值
- `(nrows, ncols)` 元组

# 实现说明
调用 nrows(ci) 和 ncols(ci) 来获取行列数。
子类型需要定义这两个辅助函数。
"""
function Base.size(ci::AbstractMatrixCI)
    return nrows(ci), ncols(ci)
end

"""
    row(ci::AbstractMatrixCI{T}, i::Int; cols=Colon()) where {T}

提取近似矩阵的第i行（或其子集）。

# 参数
- `ci`: 矩阵交叉插值对象
- `i::Int`: 行索引
- `cols`: 列选择器，默认为所有列

# 返回值
- 一维向量，包含指定行的元素
"""
function row(
    ci::AbstractMatrixCI{T},
    i::Int;
    cols::Union{AbstractVector{Int},Colon}=Colon()  # Colon()表示全部
) where {T}
    # submatrix返回2D数组，[:]将其展平为1D
    return submatrix(ci, [i], cols)[:]
end

"""
    col(ci::AbstractMatrixCI{T}, j::Int; rows=Colon()) where {T}

提取近似矩阵的第j列（或其子集）。

# 参数
- `ci`: 矩阵交叉插值对象
- `j::Int`: 列索引
- `rows`: 行选择器，默认为所有行

# 返回值
- 一维向量，包含指定列的元素
"""
function col(
    ci::AbstractMatrixCI{T},
    j::Int;
    rows::Union{AbstractVector{Int},Colon}=Colon()
) where {T}
    return submatrix(ci, rows, [j])[:]
end

"""
    Base.getindex(ci::AbstractMatrixCI{T}, rows, cols) where {T}

使用标准索引语法访问近似矩阵的子矩阵。

# 语法
```julia
ci[1:3, 2:5]  # 获取第1-3行，第2-5列
ci[:, 1:10]   # 获取所有行，前10列
```

# 注意
这实际上返回的是近似值，而非原始矩阵的精确值。
"""
function Base.getindex(
    ci::AbstractMatrixCI{T},
    rows::Union{AbstractVector{Int},Colon},
    cols::Union{AbstractVector{Int},Colon}
) where {T}
    return submatrix(ci, rows, cols)
end

"""
    Base.getindex(ci::AbstractMatrixCI{T}, i::Int, cols) where {T}

访问单行的多个列。

# 语法
```julia
ci[5, :]      # 第5行的所有列
ci[5, 1:10]   # 第5行的前10列
```
"""
function Base.getindex(
    ci::AbstractMatrixCI{T},
    i::Int,
    cols::Union{AbstractVector{Int},Colon}
) where {T}
    return row(ci, i; cols=cols)
end

"""
    Base.getindex(ci::AbstractMatrixCI{T}, rows, j::Int) where {T}

访问多行的单列。

# 语法
```julia
ci[:, 5]      # 所有行的第5列
ci[1:10, 5]   # 前10行的第5列
```
"""
function Base.getindex(
    ci::AbstractMatrixCI{T},
    rows::Union{AbstractVector{Int},Colon},
    j::Int
) where {T}
    return col(ci, j; rows=rows)
end

"""
    Base.getindex(ci::AbstractMatrixCI{T}, i::Int, j::Int) where {T}

访问单个元素。

# 语法
```julia
ci[3, 5]  # 第3行第5列的元素
```
"""
function Base.getindex(ci::AbstractMatrixCI{T}, i::Int, j::Int) where {T}
    return evaluate(ci, i, j)
end

"""
    localerror(
        ci::AbstractMatrixCI{T},
        a::AbstractMatrix{T},
        rowindices::Union{AbstractVector{Int},Colon,Int}=Colon(),
        colindices::Union{AbstractVector{Int},Colon,Int}=Colon()
    ) where {T}

计算交叉插值在指定区域的局部误差（逐元素的绝对误差）。

# 参数
- `ci`: 矩阵交叉插值对象
- `a`: 原始精确矩阵
- `rowindices`: 要检查的行索引范围，默认全部
- `colindices`: 要检查的列索引范围，默认全部

# 返回值
- 与指定区域同形状的矩阵，每个元素是 |a[i,j] - ci[i,j]|

# 数学公式
```math
\\text{error}[i,j] = |A[i,j] - \\tilde{A}[i,j]|
```
其中 A 是原始矩阵，Ã 是交叉插值近似。

# 用途
用于评估近似质量，以及寻找下一个枢轴点（选择误差最大的位置）。
"""
function localerror(
    ci::AbstractMatrixCI{T},
    a::AbstractMatrix{T},
    rowindices::Union{AbstractVector{Int},Colon,Int}=Colon(),
    colindices::Union{AbstractVector{Int},Colon,Int}=Colon()
) where {T}
    # abs.() 逐元素取绝对值
    # .- 逐元素相减
    return abs.(a[rowindices, colindices] .- ci[rowindices, colindices])
end

"""
    findnewpivot(
        ci::AbstractMatrixCI{T},
        a::AbstractMatrix{T},
        rowindices::Union{Vector{Int}}=availablerows(ci),
        colindices::Union{Vector{Int}}=availablecols(ci)
    ) where {T}

在指定区域内找到局部误差最大的位置作为新枢轴。

# 参数
- `ci`: 当前的矩阵交叉插值
- `a`: 原始精确矩阵
- `rowindices`: 候选行索引，默认为尚未使用的行
- `colindices`: 候选列索引，默认为尚未使用的列

# 返回值
- `((row, col), error)` 元组
  - `(row, col)`: 误差最大位置的索引
  - `error`: 该位置的误差值

# 抛出异常
- 如果ci已经满秩，抛出ArgumentError
- 如果候选行或列为空，抛出ArgumentError

# 算法
在所有候选位置(i,j)中，找到使|a[i,j] - ci[i,j]|最大的位置。
这是贪婪选择策略，每次添加能最大程度减少误差的枢轴。

# 示例
```julia
ci = MatrixCI(A, (1, 1))  # 初始化，第一个枢轴在(1,1)
pivot, error = findnewpivot(ci, A)  # 找到最佳的第二个枢轴
```
"""
function findnewpivot(
    ci::AbstractMatrixCI{T},
    a::AbstractMatrix{T},
    rowindices::Union{Vector{Int}}=availablerows(ci),
    colindices::Union{Vector{Int}}=availablecols(ci)
) where {T}
    # 检查是否已满秩
    if rank(ci) == minimum(size(a))
        throw(ArgumentError(
            "Cannot find a new pivot for this MatrixCrossInterpolation, as it is
            already full rank."))
    # 检查候选行是否为空
    elseif (length(rowindices) == 0)
        throw(ArgumentError(
            "Cannot find a new pivot in an empty set of rows (row_indices = $rowindices)"
        ))
    # 检查候选列是否为空
    elseif (length(colindices) == 0)
        throw(ArgumentError(
            "Cannot find a new pivot in an empty set of cols (col_indices = $rowindices)"
        ))
    end

    # 计算所有候选位置的误差
    localerrors = localerror(ci, a, rowindices, colindices)
    
    # argmax返回最大值的CartesianIndex
    ijraw = argmax(localerrors)
    
    # 将相对索引转换回原始索引
    return (rowindices[ijraw[1]], colindices[ijraw[2]]), localerrors[ijraw]
end
