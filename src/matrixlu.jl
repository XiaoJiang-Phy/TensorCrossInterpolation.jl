# ===================================================================
# matrixlu.jl - 秩揭示LU分解 (Rank-Revealing LU Decomposition)
# ===================================================================
# 这个文件实现了秩揭示LU分解(rrLU)，它是矩阵低秩近似的另一种方法。
#
# 秩揭示LU分解是标准LU分解的变体，通过完全主元选择(complete pivoting)
# 来自动检测矩阵的数值秩，并在适当的位置停止分解。
#
# 数学背景：
# 给定矩阵 A，rrLU找到排列矩阵 P, Q 和分解：
#   P * A * Q ≈ L * U
# 其中 L 是下三角矩阵，U 是上三角矩阵，秩为 r << min(m, n)。
#
# 这种分解用于：
# 1. 压缩张量列（作为SVD的快速替代）
# 2. 交叉插值中的枢轴选择
# 3. 低秩矩阵近似
# ===================================================================

# ===================================================================
# 辅助函数：子矩阵最大值搜索
# ===================================================================

"""
    submatrixargmax(f::Function, A::AbstractMatrix{T}, rows, cols; colmask, rowmask) where {T}

在子矩阵中找到使函数 f 最大的元素位置。

# 参数
- `f`: 应用于元素的函数（如 abs2 计算平方绝对值）
- `A`: 矩阵
- `rows`: 行索引范围
- `cols`: 列索引范围
- `colmask`: 列过滤函数（返回 false 的列被跳过）
- `rowmask`: 行过滤函数（返回 false 的行被跳过）

# 返回值
- `(mr, mc)` 元组：最大值的行索引和列索引

# 用途
在完全主元LU分解中找到绝对值最大的元素作为下一个枢轴。

# 算法
简单的双重循环遍历，维护当前最大值及其位置。
"""
function submatrixargmax(
    f::Function, # 实值函数
    A::AbstractMatrix{T},
    rows::Union{AbstractVector,UnitRange},
    cols::Union{AbstractVector,UnitRange};
    colmask::Function=x->true,  # 默认不过滤
    rowmask::Function=x->true
) where {T}
    # 初始化为最小可能值
    m = typemin(f(first(A)))
    
    # 验证输入
    !isempty(rows) || throw(ArgumentError("rows must not be empty"))
    !isempty(cols) || throw(ArgumentError("cols must not be empty"))
    
    # 初始化最大值位置
    mr = first(rows)
    mc = first(cols)
    
    # 验证索引范围
    rows ⊆ axes(A, 1) || throw(ArgumentError("rows ⊆ axes(A, 1) must be satified"))
    cols ⊆ axes(A, 2) || throw(ArgumentError("cols ⊆ axes(A, 2) must be satified"))
    
    # 遍历查找最大值
    @inbounds for c in cols
        if !colmask(c)
            continue  # 跳过被过滤的列
        end
        for r in rows
            if !rowmask(r)
                continue  # 跳过被过滤的行
            end
            v = f(A[r, c])
            newm = v > m  # 是否找到新的最大值
            # ifelse 是无分支的条件选择，比 if-else 更快
            m = ifelse(newm, v, m)
            mr = ifelse(newm, r, mr)
            mc = ifelse(newm, c, mc)
        end
    end
    return mr, mc
end

"""
    submatrixargmax(A::AbstractMatrix, rows, cols)

简化版本：找到子矩阵中最大元素的位置。
"""
function submatrixargmax(
    A::AbstractMatrix,
    rows::Union{AbstractVector,UnitRange},
    cols::Union{AbstractVector,UnitRange},
)
    return submatrixargmax(identity, A, rows, cols)
end

"""
    submatrixargmax(f::Function, A::AbstractMatrix, rows, cols; kwargs...)

通用版本：支持多种索引类型。
"""
function submatrixargmax(f::Function, A::AbstractMatrix, rows, cols; colmask::Function=x->true, rowmask::Function=x->true)
    # 将各种索引类型统一转换为向量或范围
    function convertarg(arg::Int, size::Int)
        return [arg]
    end

    function convertarg(arg::Colon, size::Int)
        return 1:size
    end

    function convertarg(arg::Union{AbstractVector,UnitRange}, size::Int)
        return arg
    end

    return submatrixargmax(f, A, convertarg(rows, size(A, 1)), convertarg(cols, size(A, 2)); colmask=colmask, rowmask=rowmask)
end

function submatrixargmax(A::AbstractMatrix, rows, cols)
    submatrixargmax(identity, A::AbstractMatrix, rows, cols)
end

"""
    submatrixargmax(f::Function, A::AbstractMatrix, startindex::Int; kwargs...)

从指定起始索引开始搜索。
"""
function submatrixargmax(f::Function, A::AbstractMatrix, startindex::Int; colmask::Function=x->true, rowmask::Function=x->true)
    return submatrixargmax(f, A, startindex:size(A, 1), startindex:size(A, 2); colmask=colmask, rowmask=rowmask)
end

function submatrixargmax(A::AbstractMatrix, startindex::Int)
    return submatrixargmax(identity, A, startindex:size(A, 1), startindex:size(A, 2))
end

# ===================================================================
# rrLU 类型定义
# ===================================================================

"""
    mutable struct rrLU{T}

秩揭示LU分解的数据结构。

# 数学描述
对于矩阵 A（m×n），rrLU 分解为：
```math
P ⋅ A ⋅ Q ≈ L ⋅ U
```
其中：
- P 是 m×m 的行排列矩阵
- Q 是 n×n 的列排列矩阵
- L 是 m×r 的下三角矩阵（r 是枢轴数量/数值秩）
- U 是 r×n 的上三角矩阵

# 类型参数
- `T`: 矩阵元素类型

# 字段
- `rowpermutation::Vector{Int}`: 行排列（P 的隐式表示）
  - 第 i 行对应原矩阵的第 rowpermutation[i] 行
- `colpermutation::Vector{Int}`: 列排列（Q 的隐式表示）
  - 第 j 列对应原矩阵的第 colpermutation[j] 列
- `L::Matrix{T}`: 下三角矩阵
- `U::Matrix{T}`: 上三角矩阵
- `leftorthogonal::Bool`: 是否左正交化
  - true: L 的对角线为 1，U 包含对角元素
  - false: U 的对角线为 1，L 包含对角元素
- `npivot::Int`: 枢轴数量（近似的秩）
- `error::Float64`: 最后一个未选枢轴的误差

# 重构公式
```math
A ≈ P^(-1) ⋅ L ⋅ U ⋅ Q^(-1)
```
或者用函数表示：
```julia
approx = left(lu) * right(lu)  # 自动应用排列
```

# 特性
- 自动秩检测：根据容差自动确定数值秩
- 完全主元选择：每步选择剩余矩阵中绝对值最大的元素
- 比 SVD 更快，尽管精度稍低

# 示例
```julia
A = rand(100, 50)
lu = rrlu(A; reltol=1e-10, maxrank=30)
println("数值秩: $(npivots(lu))")
approx = left(lu) * right(lu)
error = maximum(abs.(A - approx))
```
"""
mutable struct rrLU{T}
    rowpermutation::Vector{Int}   # 行排列
    colpermutation::Vector{Int}   # 列排列
    L::Matrix{T}                  # 下三角矩阵
    U::Matrix{T}                  # 上三角矩阵
    leftorthogonal::Bool          # 左正交标志
    npivot::Int                   # 枢轴数量
    error::Float64                # 最后的误差

    """
        rrLU(rowpermutation, colpermutation, L, U, leftorthogonal, npivot, error)
    
    全参数构造函数。
    """
    function rrLU(rowpermutation::Vector{Int}, colpermutation::Vector{Int}, L::Matrix{T}, U::Matrix{T}, leftorthogonal, npivot, error) where {T}
        # 验证维度一致性
        npivot == size(L, 2) || error("L must have the same number of columns as the number of pivots.")
        npivot == size(U, 1) || error("U must have the same number of rows as the number of pivots.")
        length(rowpermutation) == size(L, 1) || error("rowpermutation must have length equal to the number of pivots.")
        length(colpermutation) == size(U, 2) || error("colpermutation must have length equal to the number of pivots.")
        new{T}(rowpermutation, colpermutation, L, U, leftorthogonal, npivot, error)
    end

    """
        rrLU{T}(nrows::Int, ncols::Int; leftorthogonal::Bool=true) where {T}
    
    创建一个空的 rrLU 对象。
    """
    function rrLU{T}(nrows::Int, ncols::Int; leftorthogonal::Bool=true) where {T}
        new{T}(1:nrows, 1:ncols, zeros(nrows, 0), zeros(0, ncols), leftorthogonal, 0, NaN)
    end
end

"""
    rrLU{T}(A::AbstractMatrix{T}; leftorthogonal::Bool=true) where {T}

从矩阵创建空的 rrLU 对象（与矩阵大小匹配）。
"""
function rrLU{T}(A::AbstractMatrix{T}; leftorthogonal::Bool=true) where {T}
    rrLU{T}(size(A)...; leftorthogonal=leftorthogonal)
end

# ===================================================================
# LU分解核心算法
# ===================================================================

"""
    swaprow!(lu::rrLU{T}, A::AbstractMatrix{T}, a, b) where {T}

交换矩阵 A 的第 a 行和第 b 行，并更新排列记录。

# 说明
这是 LU 分解中的行交换操作，同时更新 `rowpermutation`。
"""
function swaprow!(lu::rrLU{T}, A::AbstractMatrix{T}, a, b) where {T}
    lurp = lu.rowpermutation
    # 交换排列记录
    lurp[a], lurp[b] = lurp[b], lurp[a]
    # 交换矩阵行
    @inbounds for j in axes(A, 2)
        A[a, j], A[b, j] = A[b, j], A[a, j]
    end
end

"""
    swapcol!(lu::rrLU{T}, A::AbstractMatrix{T}, a, b) where {T}

交换矩阵 A 的第 a 列和第 b 列，并更新排列记录。
"""
function swapcol!(lu::rrLU{T}, A::AbstractMatrix{T}, a, b) where {T}
    lucp = lu.colpermutation
    lucp[a], lucp[b] = lucp[b], lucp[a]
    @inbounds for i in axes(A, 1)
        A[i, a], A[i, b] = A[i, b], A[i, a]
    end
end

"""
    addpivot!(lu::rrLU{T}, A::AbstractMatrix{T}, newpivot) where {T}

添加一个新枢轴并执行高斯消元。

# 参数
- `lu`: rrLU 对象（会被修改）
- `A`: 工作矩阵（会被修改）
- `newpivot`: 新枢轴位置 (row, col)

# 算法
1. 将枢轴移到对角线位置（通过行列交换）
2. 归一化（根据 leftorthogonal 归一化 L 或 U）
3. 执行高斯消元更新剩余矩阵

# 高斯消元公式
A[k+1:end, k+1:end] -= A[k+1:end, k] * A[k, k+1:end]
"""
function addpivot!(lu::rrLU{T}, A::AbstractMatrix{T}, newpivot) where {T}
    k = lu.npivot += 1  # 增加枢轴计数
    
    # 将枢轴移到位置 (k, k)
    swaprow!(lu, A, k, newpivot[1])
    swapcol!(lu, A, k, newpivot[2])

    # 归一化
    if lu.leftorthogonal
        # L 的对角线为 1，归一化下方元素
        A[k+1:end, k] ./= A[k, k]
    else
        # U 的对角线为 1，归一化右方元素
        A[k, k+1:end] ./= A[k, k]
    end

    # 执行高斯消元（手动实现 BLAS 操作以提高性能）
    # A[k+1:end, k+1:end] -= A[k+1:end, k] * A[k, k+1:end]'
    x = @view(A[k+1:end, k])
    y = @view(A[k, k+1:end])
    A = @view(A[k+1:end, k+1:end])
    @inbounds for j in eachindex(axes(A, 2), y)
        for i in eachindex(axes(A, 1), x)
            A[i, j] -= x[i] * y[j]
        end
    end
    nothing
end

# 辅助函数：计算已选择的行/列数
_count_cols_selected(lu, cols)::Int = sum([c ∈ lu.colpermutation[1:lu.npivot] for c ∈ cols])
_count_rows_selected(lu, rows)::Int = sum([r ∈ lu.rowpermutation[1:lu.npivot] for r ∈ rows])

"""
    _optimizerrlu!(lu::rrLU{T}, A::AbstractMatrix{T}; maxrank, reltol, abstol) where {T}

执行秩揭示 LU 分解的核心优化循环（内部函数）。

# 参数
- `lu`: rrLU 对象（会被修改）
- `A`: 工作矩阵（会被修改）
- `maxrank`: 最大秩
- `reltol`: 相对容差
- `abstol`: 绝对容差

# 算法
1. 在剩余子矩阵中找到绝对值最大的元素
2. 检查是否满足停止条件
3. 添加枢轴并执行消元
4. 重复直到满足条件
"""
function _optimizerrlu!(
    lu::rrLU{T},
    A::AbstractMatrix{T};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0
) where {T}
    maxrank = min(maxrank, size(A, 1), size(A, 2))
    maxerror = 0.0
    
    while lu.npivot < maxrank
        k = lu.npivot + 1
        # 在剩余矩阵中找最大元素
        newpivot = submatrixargmax(abs2, A, k)
        lu.error = abs(A[newpivot[1], newpivot[2]])
        
        # 检查停止条件（至少添加一个枢轴以确保 L*U 有定义）
        if (abs(lu.error) < reltol * maxerror || abs(lu.error) < abstol) && lu.npivot > 0
            break
        end
        
        maxerror = max(maxerror, lu.error)
        addpivot!(lu, A, newpivot)
    end

    # 提取 L 和 U 矩阵
    lu.L = tril(@view A[:, 1:lu.npivot])  # 下三角部分
    lu.U = triu(@view A[1:lu.npivot, :])  # 上三角部分
    
    # 检查 NaN
    if any(isnan.(lu.L))
        error("lu.L contains NaNs")
    end
    if any(isnan.(lu.U))
        error("lu.U contains NaNs")
    end
    
    # 设置对角线
    if lu.leftorthogonal
        lu.L[diagind(lu.L)] .= one(T)  # L 对角线设为 1
    else
        lu.U[diagind(lu.U)] .= one(T)  # U 对角线设为 1
    end

    # 满秩时误差为 0
    if lu.npivot >= minimum(size(A))
        lu.error = 0
    end

    nothing
end

# ===================================================================
# 公共接口函数
# ===================================================================

"""
    rrlu!(A::AbstractMatrix{T}; maxrank, reltol, abstol, leftorthogonal)::rrLU{T} where {T}

原地秩揭示 LU 分解（会修改输入矩阵）。

# 参数
- `A`: 输入矩阵（会被修改）
- `maxrank`: 最大秩，默认无限制
- `reltol`: 相对容差，默认 1e-14
- `abstol`: 绝对容差，默认 0.0
- `leftorthogonal`: 是否左正交化，默认 true

# 返回值
- `rrLU{T}` 对象

# 警告
此函数会修改输入矩阵 A！如需保留原矩阵，使用 `rrlu` 而非 `rrlu!`。
"""
function rrlu!(
    A::AbstractMatrix{T};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true
)::rrLU{T} where {T}
    lu = rrLU{T}(A, leftorthogonal=leftorthogonal)
    _optimizerrlu!(lu, A; maxrank=maxrank, reltol=reltol, abstol=abstol)
    return lu
end

"""
    rrlu(A::AbstractMatrix{T}; maxrank, reltol, abstol, leftorthogonal)::rrLU{T} where {T}

秩揭示 LU 分解（不修改输入矩阵）。

# 参数
- `A`: 输入矩阵
- `maxrank`: 最大秩
- `reltol`: 相对容差
- `abstol`: 绝对容差
- `leftorthogonal`: 是否左正交化

# 返回值
- `rrLU{T}` 对象

# 示例
```julia
A = rand(100, 50)
lu = rrlu(A; reltol=1e-10)
rank_A = npivots(lu)
println("数值秩: \$rank_A")
```
"""
function rrlu(
    A::AbstractMatrix{T};
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true
)::rrLU{T} where {T}
    return rrlu!(copy(A); maxrank, reltol, abstol, leftorthogonal)
end

"""
    arrlu(::Type{ValueType}, f, matrixsize, I0, J0; kwargs...) where {ValueType}

自适应秩揭示 LU 分解（不需要完整矩阵）。

# 背景
当矩阵很大但低秩时，可以只采样部分元素来进行分解。
这个函数使用"车象搜索"(rook pivoting)策略来高效找到枢轴。

# 参数
- `ValueType`: 元素类型
- `f`: 矩阵元素函数 f(i, j) 返回 A[i, j]
- `matrixsize`: 矩阵大小 (m, n)
- `I0, J0`: 初始行/列索引集
- `maxrank`: 最大秩
- `reltol, abstol`: 容差
- `numrookiter`: 车象搜索迭代次数
- `usebatcheval`: 是否使用批量求值

# 返回值
- `rrLU{ValueType}` 对象

# 算法（车象搜索）
1. 随机选择初始行/列
2. 交替在行和列方向搜索最大元素
3. 收敛后扩展到完整秩
"""
function arrlu(
    ::Type{ValueType},
    f,
    matrixsize::Tuple{Int,Int},
    I0::AbstractVector{Int}=Int[],
    J0::AbstractVector{Int}=Int[];
    maxrank::Int=typemax(Int),
    reltol::Number=1e-14,
    abstol::Number=0.0,
    leftorthogonal::Bool=true,
    numrookiter::Int=5,
    usebatcheval::Bool=false
)::rrLU{ValueType} where {ValueType}
    lu = rrLU{ValueType}(matrixsize...; leftorthogonal)
    islowrank = false
    maxrank = min(maxrank, matrixsize...)

    # 选择批量求值函数
    _batchf = usebatcheval ? f : ((x, y) -> f.(x, y'))

    while true
        # 扩展初始索引集
        if leftorthogonal
            pushrandomsubset!(J0, 1:matrixsize[2], max(1, length(J0)))
        else
            pushrandomsubset!(I0, 1:matrixsize[1], max(1, length(I0)))
        end

        # 车象搜索迭代
        for rookiter in 1:numrookiter
            colmove = (iseven(rookiter) == leftorthogonal)
            
            # 采样子矩阵
            submatrix = if colmove
                _batchf(I0, lu.colpermutation)
            else
                _batchf(lu.rowpermutation, J0)
            end
            
            # 对子矩阵进行 LU 分解
            lu.npivot = 0
            _optimizerrlu!(lu, submatrix; maxrank, reltol, abstol)
            islowrank |= npivots(lu) < minimum(size(submatrix))
            
            # 检查收敛
            if rowindices(lu) == I0 && colindices(lu) == J0
                break
            end

            J0 = colindices(lu)
            I0 = rowindices(lu)
        end

        if islowrank || length(I0) >= maxrank
            break
        end
    end

    # 扩展 L 矩阵到完整行
    if size(lu.L, 1) < matrixsize[1]
        I2 = setdiff(1:matrixsize[1], I0)
        lu.rowpermutation = vcat(I0, I2)
        L2 = _batchf(I2, J0)
        cols2Lmatrix!(L2, (@view lu.U[1:lu.npivot, 1:lu.npivot]), leftorthogonal)
        lu.L = vcat((@view lu.L[1:lu.npivot, 1:lu.npivot]), L2)
    end

    # 扩展 U 矩阵到完整列
    if size(lu.U, 2) < matrixsize[2]
        J2 = setdiff(1:matrixsize[2], J0)
        lu.colpermutation = vcat(J0, J2)
        U2 = _batchf(I0, J2)
        rows2Umatrix!(U2, (@view lu.L[1:lu.npivot, 1:lu.npivot]), leftorthogonal)
        lu.U = hcat((@view lu.U[1:lu.npivot, 1:lu.npivot]), U2)
    end

    return lu
end

"""
    rrlu(::Type{ValueType}, f, matrixsize, I0, J0; pivotsearch=:full, kwargs...) where {ValueType}

基于函数的秩揭示 LU 分解。

# 参数
- `pivotsearch`: 枢轴搜索策略
  - `:full`: 完全枢轴选择（采样整个矩阵）
  - `:rook`: 车象搜索（只采样部分元素）

# 示例
```julia
# 定义矩阵函数
f(i, j) = 1.0 / (i + j - 1)  # Hilbert 矩阵

# 使用完全搜索
lu = rrlu(Float64, f, (100, 100); pivotsearch=:full)

# 使用车象搜索（更快但可能不够精确）
lu = rrlu(Float64, f, (100, 100); pivotsearch=:rook)
```
"""
function rrlu(
    ::Type{ValueType},
    f,
    matrixsize::Tuple{Int,Int},
    I0::AbstractVector{Int}=Int[],
    J0::AbstractVector{Int}=Int[];
    pivotsearch=:full,
    kwargs...
)::rrLU{ValueType} where {ValueType}
    if pivotsearch === :rook
        return arrlu(ValueType, f, matrixsize, I0, J0; kwargs...)
    elseif pivotsearch === :full
        # 构建完整矩阵并分解
        A = f.(1:matrixsize[1], collect(1:matrixsize[2])')
        return rrlu!(A; kwargs...)
    else
        throw(ArgumentError("Unknown pivot search strategy $pivotsearch. Choose between :rook and :full."))
    end
end

# ===================================================================
# 辅助矩阵转换函数
# ===================================================================

"""
    cols2Lmatrix!(C::AbstractMatrix, P::AbstractMatrix, leftorthogonal::Bool)

将列矩阵转换为 L 矩阵格式（内部函数）。

# 参数
- `C`: 要转换的列矩阵（会被修改）
- `P`: 枢轴矩阵（用于归一化）
- `leftorthogonal`: 正交化方向
"""
function cols2Lmatrix!(C::AbstractMatrix, P::AbstractMatrix, leftorthogonal::Bool)
    if size(C, 2) != size(P, 2)
        throw(DimensionMismatch("C and P matrices must have same number of columns in `cols2Lmatrix!`."))
    elseif size(P, 1) != size(P, 2)
        throw(DimensionMismatch("P matrix must be square in `cols2Lmatrix!`."))
    end

    for k in axes(P, 1)
        C[:, k] ./= P[k, k]
        # 高斯消元
        x = @view C[:, k]
        y = @view P[k, k+1:end]
        C̃ = @view C[:, k+1:end]
        @inbounds for j in eachindex(axes(C̃, 2), y)
            for i in eachindex(axes(C̃, 1), x)
                C̃[i, j] -= x[i] * y[j]
            end
        end
    end
    return C
end

"""
    rows2Umatrix!(R::AbstractMatrix, P::AbstractMatrix, leftorthogonal::Bool)

将行矩阵转换为 U 矩阵格式（内部函数）。
"""
function rows2Umatrix!(R::AbstractMatrix, P::AbstractMatrix, leftorthogonal::Bool)
    if size(R, 1) != size(P, 1)
        throw(DimensionMismatch("R and P matrices must have same number of rows in `rows2Umatrix!`."))
    elseif size(P, 1) != size(P, 2)
        throw(DimensionMismatch("P matrix must be square in `rows2Umatrix!`."))
    end

    for k in axes(P, 1)
        R[k, :] ./= P[k, k]
        # 高斯消元
        x = @view P[k+1:end, k]
        y = @view R[k, :]
        R̃ = @view R[k+1:end, :]
        @inbounds for j in eachindex(axes(R̃, 2), y)
            for i in eachindex(axes(R̃, 1), x)
                R̃[i, j] -= x[i] * y[j]
            end
        end
    end
    return R
end

# ===================================================================
# 访问器函数
# ===================================================================

"""
    size(lu::rrLU{T}) where {T}

获取原矩阵的大小 (m, n)。
"""
function size(lu::rrLU{T}) where {T}
    return size(lu.L, 1), size(lu.U, 2)
end

"""
    size(lu::rrLU{T}, dim) where {T}

获取原矩阵在指定维度的大小。
"""
function size(lu::rrLU{T}, dim) where {T}
    if dim == 1
        return size(lu.L, 1)
    elseif dim == 2
        return size(lu.U, 2)
    else
        return 1
    end
end

"""
    left(lu::rrLU{T}; permute=true) where {T}

获取左因子矩阵（L 矩阵）。

# 参数
- `permute`: 是否应用行排列

# 返回值
- 如果 permute=true: 按原矩阵行顺序排列的 L 矩阵
- 如果 permute=false: 按 LU 分解顺序的 L 矩阵
"""
function left(lu::rrLU{T}; permute=true) where {T}
    if permute
        l = Matrix{T}(undef, size(lu.L)...)
        l[lu.rowpermutation, :] = lu.L
        return l
    else
        return lu.L
    end
end

"""
    right(lu::rrLU{T}; permute=true) where {T}

获取右因子矩阵（U 矩阵）。

# 参数
- `permute`: 是否应用列排列

# 返回值
- 如果 permute=true: 按原矩阵列顺序排列的 U 矩阵
- 如果 permute=false: 按 LU 分解顺序的 U 矩阵
"""
function right(lu::rrLU{T}; permute=true) where {T}
    if permute
        u = Matrix{T}(undef, size(lu.U)...)
        u[:, lu.colpermutation] = lu.U
        return u
    else
        return lu.U
    end
end

"""
    diag(lu::rrLU{T}) where {T}

获取枢轴矩阵的对角线元素。

# 返回值
- 对角线元素向量（枢轴值）
"""
function diag(lu::rrLU{T}) where {T}
    if lu.leftorthogonal
        return diag(lu.U[1:lu.npivot, 1:lu.npivot])
    else
        return diag(lu.L[1:lu.npivot, 1:lu.npivot])
    end
end

"""
    rowindices(lu::rrLU{T}) where {T}

获取枢轴行索引（在原矩阵中的位置）。
"""
function rowindices(lu::rrLU{T}) where {T}
    return lu.rowpermutation[1:lu.npivot]
end

"""
    colindices(lu::rrLU{T}) where {T}

获取枢轴列索引（在原矩阵中的位置）。
"""
function colindices(lu::rrLU{T}) where {T}
    return lu.colpermutation[1:lu.npivot]
end

"""
    npivots(lu::rrLU{T}) where {T}

获取枢轴数量（数值秩）。
"""
function npivots(lu::rrLU{T}) where {T}
    return lu.npivot
end

"""
    pivoterrors(lu::rrLU{T}) where {T}

获取枢轴误差数组。

# 返回值
- 向量，包含每个枢轴的对角元素绝对值，最后一个元素是剩余误差
"""
function pivoterrors(lu::rrLU{T}) where {T}
    return vcat(abs.(diag(lu)), lu.error)
end

"""
    lastpivoterror(lu::rrLU{T})::Float64 where {T}

获取最后一个枢轴的误差。

# 说明
对于满秩矩阵，返回 0。
"""
function lastpivoterror(lu::rrLU{T})::Float64 where {T}
    return lu.error
end

# ===================================================================
# 线性方程组求解
# ===================================================================

"""
    solve(L::Matrix{T}, U::Matrix{T}, b::Matrix{T}) where {T}

求解 (L * U) * x = b。

# 参数
- `L`: 下三角矩阵
- `U`: 上三角矩阵
- `b`: 右侧向量/矩阵

# 返回值
- 解 x

# 算法
1. 前向替换：求解 L * y = b
2. 后向替换：求解 U * x = y

# 注意
此实现未进行性能优化，适用于小规模问题。
"""
function solve(L::Matrix{T}, U::Matrix{T}, b::Matrix{T}) where{T}
    N1, N2, N3 = size(L, 1), size(L, 2), size(U, 2)
    M = size(b, 2)
    
    # 前向替换：求解 L * y = b
    y = zeros(T, N2, M)
    for i = 1:N2
        y[i, :] .= b[i, :]
        for k in 1:M
            for j in 1:i-1
                y[i, k] -= L[i, j] * y[j, k]
            end
        end
        y[i, :] ./= L[i, i]
    end

    # 后向替换：求解 U * x = y
    x = zeros(T, N3, M)
    for i = N3:-1:1
        x[i, :] .= y[i, :]
        for k in 1:M
            for j = i+1:N3
                x[i, k] -= U[i, j] * x[j, k]
            end
        end
        x[i, :] ./= U[i, i]
    end

    return x
end

"""
    Base.:\\(A::rrLU{T}, b::AbstractMatrix{T}) where {T}

使用 LU 分解求解 A * x = b。

# 说明
重载反斜杠运算符，使 `x = lu \\ b` 可以直接使用。

# 要求
- 矩阵必须是方阵
- 矩阵必须是满秩的

# 示例
```julia
A = rand(10, 10)
b = rand(10, 3)
lu = rrlu(A)
x = lu \\ b
# 验证: A * x ≈ b
```
"""
function Base.:\(A::rrLU{T}, b::AbstractMatrix{T}) where{T}
    size(A, 1) == size(A, 2) || error("Matrix must be square.")
    A.npivot == size(A, 1) || error("rank-deficient matrix is not supportred!")
    
    # 应用行排列
    b_perm = b[A.rowpermutation, :]
    # 求解
    x_perm = solve(A.L, A.U, b_perm)
    # 应用列排列的逆
    x = similar(x_perm)
    for i in 1:size(x, 1)
        x[A.colpermutation[i], :] .= x_perm[i, :]
    end
    return x
end

"""
    Base.transpose(A::rrLU{T}) where {T}

获取 LU 分解的转置。

# 返回值
- 新的 rrLU 对象，表示 Aᵀ 的分解

# 数学说明
如果 A = L * U，则 Aᵀ = Uᵀ * Lᵀ
"""
function Base.transpose(A::rrLU{T}) where{T}
    return rrLU(
        A.colpermutation, A.rowpermutation,
        Matrix(transpose(A.U)), Matrix(transpose(A.L)), 
        !A.leftorthogonal, A.npivot, A.error)
end