# ===================================================================
# matrixci.jl - 矩阵交叉插值
# ===================================================================
# 这个文件实现了MatrixCI类型，它是矩阵交叉插值的核心实现。
#
# 矩阵交叉插值(Cross Approximation)是一种低秩矩阵近似方法：
#   A ≈ A[:, J] * (A[I, J])^(-1) * A[I, :]
# 其中 I 和 J 分别是选中的行索引和列索引集合。
#
# 这种分解只需要存储 m×r + r×n 个元素（而不是 m×n），
# 其中 r = |I| = |J| 是枢轴数量（近似的秩）。
# ===================================================================

# ===================================================================
# 辅助函数：数值稳定的矩阵运算
# ===================================================================

"""
    AtimesBinv(A::AbstractVecOrMat, B::AbstractMatrix)

计算矩阵乘积 A * B⁻¹，使用QR分解保证数值稳定性。

# 数学背景
直接计算 A * inv(B) 在 B 病态(ill-conditioned)时数值不稳定。
通过 QR 分解可以避免这个问题。

# 算法
1. 将 A 和 B 垂直堆叠: AB = [A; B]
2. 对 AB 进行 QR 分解: AB = Q * R
3. 提取 QA = Q[1:m, :] 和 QB = Q[m+1:end, :]
4. 返回 QA * inv(QB)

# 参数
- `A`: 矩阵或向量，形状 (m × n)
- `B`: 方阵，形状 (n × n)

# 返回值
- A * B⁻¹，形状 (m × n)

# 为什么有效
由于 [A; B] = Q * R，有 A = QA * R 和 B = QB * R
因此 A * B⁻¹ = QA * R * R⁻¹ * QB⁻¹ = QA * QB⁻¹
"""
function AtimesBinv(A::AbstractVecOrMat, B::AbstractMatrix)
    m, n = size(A)
    # vcat 垂直连接矩阵
    AB = vcat(A, B)
    # QR 分解
    decomposition = LinearAlgebra.qr(AB)
    # 提取 Q 矩阵的各部分
    QA = decomposition.Q[1:m, 1:n]        # A 对应的 Q 部分
    QB = decomposition.Q[(m+1):end, 1:n]  # B 对应的 Q 部分
    # 计算最终结果
    return QA * inv(QB)
end

"""
    AinvtimesB(A::AbstractMatrix, B::AbstractVecOrMat)

计算矩阵乘积 A⁻¹ * B，使用QR分解保证数值稳定性。

# 数学背景
利用恒等式：A⁻¹ * B = (Bᵀ * A⁻ᵀ)ᵀ = (Bᵀ * (Aᵀ)⁻¹)ᵀ
因此可以使用 AtimesBinv 的转置版本。

# 参数
- `A`: 方阵，形状 (n × n)
- `B`: 矩阵或向量，形状 (n × m)

# 返回值
- A⁻¹ * B，形状 (n × m)
"""
function AinvtimesB(A::AbstractMatrix, B::AbstractVecOrMat)
    # 利用 (A⁻¹B)ᵀ = Bᵀ(A⁻¹)ᵀ = Bᵀ(Aᵀ)⁻¹
    return AtimesBinv(B', A')'
end

# ===================================================================
# MatrixCI 类型定义
# ===================================================================

"""
    mutable struct MatrixCI{T} <: AbstractMatrixCI{T}

矩阵交叉插值的数据结构。

# 数学描述
给定一个 m×n 的矩阵 A，交叉插值找到行索引集 I 和列索引集 J，
使得：A ≈ A[:, J] * (A[I, J])⁻¹ * A[I, :]

其中：
- A[:, J] 称为"枢轴列"(pivot columns)，大小 m × r
- A[I, :] 称为"枢轴行"(pivot rows)，大小 r × n
- A[I, J] 是"枢轴矩阵"(pivot matrix)，大小 r × r
- r = |I| = |J| 是枢轴数量

# 类型参数
- `T`: 矩阵元素的类型

# 字段
- `rowindices::Vector{Int}`: 选中的行索引集合 I（在xfac代码中称为Iset）
- `colindices::Vector{Int}`: 选中的列索引集合 J（在xfac代码中称为Jset）
- `pivotcols::Matrix{T}`: 枢轴列 A[:, J]，大小 (nrows, npivots)
- `pivotrows::Matrix{T}`: 枢轴行 A[I, :]，大小 (npivots, ncols)

# 重构公式
近似矩阵 = pivotcols * (pivotmatrix)⁻¹ * pivotrows
        = leftmatrix * pivotrows
其中 leftmatrix = pivotcols * (pivotmatrix)⁻¹

# 示例
```julia
A = rand(100, 50)
ci = MatrixCI(A, (argmax(abs.(A))))  # 从最大元素开始
for i in 1:10
    addpivot!(ci, A)  # 添加更多枢轴
end
approx = Matrix(ci)  # 重构近似矩阵
```
"""
mutable struct MatrixCI{T} <: AbstractMatrixCI{T}
    "行索引集 I（选中的枢轴行）"
    rowindices::Vector{Int}
    "列索引集 J（选中的枢轴列）"
    colindices::Vector{Int}
    "枢轴列 A[:, J]，在xfac代码中称为C，在TCI论文中记为 A(𝕀, 𝒥)"
    pivotcols::Matrix{T}
    "枢轴行 A[I, :]，在xfac代码中称为R，在TCI论文中记为 A(ℐ, 𝕁)"
    pivotrows::Matrix{T}

    """
        MatrixCI(::Type{T}, nrows::Int, ncols::Int) where {T<:Number}
    
    创建一个空的MatrixCI（没有枢轴）。
    
    # 参数
    - `T`: 元素类型
    - `nrows`: 原矩阵行数
    - `ncols`: 原矩阵列数
    """
    function MatrixCI(
        ::Type{T},
        nrows::Int, ncols::Int
    ) where {T<:Number}
        # zeros(nrows, 0) 创建 nrows×0 的空矩阵
        return new{T}([], [], zeros(nrows, 0), zeros(0, ncols))
    end

    """
        MatrixCI(rowindices, colindices, pivotcols, pivotrows)
    
    从已有数据创建MatrixCI。
    """
    function MatrixCI(
        rowindices::AbstractVector{Int}, colindices::AbstractVector{Int},
        pivotcols::AbstractMatrix{T}, pivotrows::AbstractMatrix{T}
    ) where {T<:Number}
        return new{T}(rowindices, colindices, pivotcols, pivotrows)
    end
end

"""
    MatrixCI(A::AbstractMatrix{T}, firstpivot) where {T<:Number}

从矩阵和第一个枢轴创建MatrixCI。

# 参数
- `A`: 原始矩阵
- `firstpivot`: 第一个枢轴的位置，如 (i, j) 或 CartesianIndex(i, j)

# 示例
```julia
A = rand(10, 10)
ci = MatrixCI(A, (5, 3))  # 第一个枢轴在(5, 3)
ci = MatrixCI(A, argmax(abs.(A)))  # 第一个枢轴在最大元素
```
"""
function MatrixCI(
    A::AbstractMatrix{T}, firstpivot
) where {T<:Number}
    return MatrixCI(
        [firstpivot[1]], [firstpivot[2]],       # 行列索引
        A[:, [firstpivot[2]]],                   # 第一个枢轴列
        A[[firstpivot[1]], :]                    # 第一个枢轴行
    )
end

# ===================================================================
# 访问器函数
# ===================================================================

"""
    Iset(ci::MatrixCI{T}) where {T}

获取行索引集（枢轴行的索引）。
"""
function Iset(ci::MatrixCI{T}) where {T}
    return ci.rowindices
end

"""
    Jset(ci::MatrixCI{T}) where {T}

获取列索引集（枢轴列的索引）。
"""
function Jset(ci::MatrixCI{T}) where {T}
    return ci.colindices
end

"""
    nrows(ci::MatrixCI)

获取原矩阵的行数。
"""
function nrows(ci::MatrixCI)
    return size(ci.pivotcols, 1)
end

"""
    ncols(ci::MatrixCI)

获取原矩阵的列数。
"""
function ncols(ci::MatrixCI)
    return size(ci.pivotrows, 2)
end

"""
    pivotmatrix(ci::MatrixCI{T}) where {T}

获取枢轴矩阵 A[I, J]。

# 返回值
- r × r 的方阵，其中 r 是枢轴数量

# 说明
枢轴矩阵是 pivotcols 的子矩阵，只取枢轴行的部分。
"""
function pivotmatrix(ci::MatrixCI{T}) where {T}
    return ci.pivotcols[ci.rowindices, :]
end

"""
    leftmatrix(ci::MatrixCI{T}) where {T}

计算左因子矩阵 A[:, J] * (A[I, J])⁻¹。

# 返回值
- m × r 的矩阵

# 数学说明
近似矩阵 = leftmatrix * pivotrows
使用 AtimesBinv 保证数值稳定性。
"""
function leftmatrix(ci::MatrixCI{T}) where {T}
    return AtimesBinv(ci.pivotcols, pivotmatrix(ci))
end

"""
    rightmatrix(ci::MatrixCI{T}) where {T}

计算右因子矩阵 (A[I, J])⁻¹ * A[I, :]。

# 返回值
- r × n 的矩阵
"""
function rightmatrix(ci::MatrixCI{T}) where {T}
    return AinvtimesB(pivotmatrix(ci), ci.pivotrows)
end

"""
    availablerows(ci::MatrixCI{T}) where {T}

获取尚未被选为枢轴的行索引。

# 返回值
- 未使用行索引的数组
"""
function availablerows(ci::MatrixCI{T}) where {T}
    # setdiff(A, B) 返回在A中但不在B中的元素
    return setdiff(1:nrows(ci), ci.rowindices)
end

"""
    availablecols(ci::MatrixCI{T}) where {T}

获取尚未被选为枢轴的列索引。
"""
function availablecols(ci::MatrixCI{T}) where {T}
    return setdiff(1:ncols(ci), ci.colindices)
end

"""
    rank(ci::MatrixCI{T}) where {T}

获取当前的枢轴数量（近似的秩）。
"""
function rank(ci::MatrixCI{T}) where {T}
    return length(ci.rowindices)
end

"""
    Base.isempty(ci::MatrixCI)

检查是否没有枢轴。
"""
function Base.isempty(ci::MatrixCI)
    return Base.isempty(ci.colindices)
end

"""
    firstpivotvalue(ci::MatrixCI{T}) where {T}

获取第一个枢轴位置的值。
"""
function firstpivotvalue(ci::MatrixCI{T}) where {T}
    return isempty(ci) ? 1.0 : ci.pivotcols[ci.rowindices[1], 1]
end

# ===================================================================
# 求值函数
# ===================================================================

"""
    evaluate(ci::MatrixCI{T}, i::Int, j::Int) where {T}

计算位置(i, j)处的近似值。

# 数学公式
Ã[i, j] = leftmatrix[i, :] ⋅ pivotrows[:, j]

# 参数
- `i`: 行索引
- `j`: 列索引

# 返回值
- 近似值（标量）
"""
function evaluate(ci::MatrixCI{T}, i::Int, j::Int) where {T}
    if isempty(ci)
        return T(0)  # 空的CI返回0
    else
        # dot 计算向量点积
        return dot(leftmatrix(ci)[i, :], ci.pivotrows[:, j])
    end
end

# 辅助函数：处理Colon()的长度
function _lengthordefault(c::Colon, default)
    return default
end

function _lengthordefault(c, default)
    return length(c)
end

"""
    submatrix(ci::MatrixCI{T}, rows, cols) where {T}

提取近似矩阵的子矩阵。

# 参数
- `rows`: 行选择器（数组或Colon）
- `cols`: 列选择器（数组或Colon）

# 返回值
- 近似矩阵的子矩阵
"""
function submatrix(
    ci::MatrixCI{T},
    rows::Union{AbstractVector{Int},Colon,Int},
    cols::Union{AbstractVector{Int},Colon,Int}
) where {T}
    if isempty(ci)
        return zeros(
            T,
            _lengthordefault(rows, nrows(ci)),
            _lengthordefault(cols, ncols(ci)))
    else
        # 矩阵乘法：leftmatrix[rows, :] * pivotrows[:, cols]
        return leftmatrix(ci)[rows, :] * ci.pivotrows[:, cols]
    end
end

"""
    Base.isapprox(lhs::MatrixCI{T}, rhs::MatrixCI{T}) where {T}

比较两个MatrixCI是否近似相等。
"""
function Base.isapprox(
    lhs::MatrixCI{T}, rhs::MatrixCI{T}
) where {T}
    return (lhs.colindices == rhs.colindices) &&
           (lhs.rowindices == rhs.rowindices) &&
           Base.isapprox(lhs.pivotcols, rhs.pivotcols) &&
           Base.isapprox(lhs.pivotrows, rhs.pivotrows)
end

"""
    Base.Matrix(ci::MatrixCI{T}) where {T}

将MatrixCI转换为普通矩阵（重构近似）。

# 返回值
- 完整的 m × n 近似矩阵
"""
function Base.Matrix(ci::MatrixCI{T}) where {T}
    return leftmatrix(ci) * ci.pivotrows
end

# ===================================================================
# 枢轴添加函数
# ===================================================================

"""
    addpivotrow!(ci::MatrixCI{T}, a::AbstractMatrix{T}, rowindex::Int) where {T}

添加一个新的枢轴行。

# 参数
- `ci`: MatrixCI对象（会被修改）
- `a`: 原始矩阵
- `rowindex`: 要添加的行索引

# 异常
- `DimensionMismatch`: 如果矩阵尺寸不匹配
- `BoundsError`: 如果索引超出范围
- `ArgumentError`: 如果该行已经是枢轴
"""
function addpivotrow!(
    ci::MatrixCI{T},
    a::AbstractMatrix{T},
    rowindex::Int
) where {T}
    # 验证尺寸
    if size(a) != size(ci)
        throw(DimensionMismatch(
            "This matrix doesn't match the MatrixCrossInterpolation object. Their sizes
            mismatch: $(size(a)) != $(size(ci))."))
    elseif (rowindex < 0) || (rowindex > nrows(ci))
        throw(BoundsError(
            "Cannot add row at row index $rowindex: it's out of bounds for a
            $(nrows(ci)) * $(ncols(ci)) matrix."))
    elseif rowindex in ci.rowindices
        throw(ArgumentError(
            "Cannot add row $rowindex: it already has a pivot."))
    end

    # 提取该行并添加
    row = transpose(a[rowindex, :])  # 转置为行向量
    ci.pivotrows = vcat(ci.pivotrows, row)  # 垂直连接
    push!(ci.rowindices, rowindex)
end

"""
    addpivotcol!(ci::MatrixCI{T}, a::AbstractMatrix{T}, colindex::Int) where {T}

添加一个新的枢轴列。

# 参数
- `ci`: MatrixCI对象（会被修改）
- `a`: 原始矩阵
- `colindex`: 要添加的列索引
"""
function addpivotcol!(
    ci::MatrixCI{T},
    a::AbstractMatrix{T},
    colindex::Int
) where {T}
    if size(a) != size(ci)
        throw(DimensionMismatch(
            "This matrix doesn't match the MatrixCrossInterpolation object. Their sizes
            mismatch: $(size(a)) != $(size(ci))."))
    elseif (colindex < 0) || (colindex > ncols(ci))
        throw(BoundsError(
            "Cannot add col at col index $colindex: it's out of bounds for a
            $(nrows(ci)) * $(ncols(ci)) matrix."))
    elseif colindex in ci.colindices
        throw(ArgumentError(
            "Cannot add column $colindex because it already has a pivot."))
    end
    
    col = a[:, colindex]
    ci.pivotcols = hcat(ci.pivotcols, col)  # 水平连接
    push!(ci.colindices, colindex)
end

"""
    addpivot!(ci::MatrixCI{T}, a::AbstractMatrix{T}, pivotindices) where {T}

在指定位置添加一个新枢轴（同时添加行和列）。

# 参数
- `ci`: MatrixCI对象
- `a`: 原始矩阵
- `pivotindices`: 枢轴位置，如 (i, j), CartesianIndex(i, j), 或 i => j
"""
function addpivot!(
    ci::MatrixCI{T},
    a::AbstractMatrix{T},
    pivotindices::Union{CartesianIndex{2},Tuple{Int,Int},Pair{Int,Int}}
) where {T}
    i = pivotindices[1]
    j = pivotindices[2]

    # 参数验证
    if size(a) != size(ci)
        throw(DimensionMismatch(
            "This matrix doesn't match the MatrixCrossInterpolation object. Their sizes
            mismatch: $(size(a)) != $(size(ci))."))
    elseif (i < 0) || (i > nrows(ci)) || (j < 0) || (j > ncols(ci))
        throw(BoundsError(
            "Cannot add a pivot at indices ($i, $j): These indices are out of bounds for a
            $(nrows(ci)) * $(ncols(ci)) matrix."))
    elseif i in ci.rowindices
        throw(ArgumentError(
            "Cannot add a pivot at indices ($i, $j) because row $i already has a pivot."))
    elseif j in ci.colindices
        throw(ArgumentError(
            "Cannot add a pivot at indices ($i, $j) because col $j already has a pivot."))
    end

    # 添加行和列
    addpivotrow!(ci, a, pivotindices[1])
    addpivotcol!(ci, a, pivotindices[2])
end

"""
    addpivot!(ci::MatrixCI{T}, a::AbstractMatrix{T}) where {T}

自动选择最大误差位置作为新枢轴。

# 说明
通过 findnewpivot 找到误差最大的位置，然后添加该枢轴。
这是贪婪算法：每次添加能最大程度减少误差的枢轴。
"""
function addpivot!(
    ci::MatrixCI{T},
    a::AbstractMatrix{T}
) where {T}
    addpivot!(ci, a, findnewpivot(ci, a)[1])
end

# ===================================================================
# 矩阵交叉插值主函数
# ===================================================================

"""
    crossinterpolate(a::AbstractMatrix{T}; tolerance=1e-6, maxiter=200, firstpivot=argmax(abs.(a))) where {T}

对矩阵进行交叉插值。

# 数学描述
找到行索引集 I 和列索引集 J，使得：
A(1:m, 1:n) ≈ A(1:m, J) * (A(I, J))⁻¹ * A(I, 1:n)

# 参数
- `a`: 要插值的矩阵
- `tolerance=1e-6`: 误差容差，当最大局部误差小于此值时停止
- `maxiter=200`: 最大迭代次数
- `firstpivot`: 第一个枢轴位置，默认为最大绝对值位置

# 返回值
- `MatrixCI{T}` 对象

# 算法（贪婪交叉近似）
1. 从 firstpivot 开始
2. 计算当前近似与原矩阵的误差
3. 在误差最大的位置添加新枢轴
4. 重复直到误差小于 tolerance 或达到 maxiter

# 示例
```julia
A = rand(100, 50)
ci = crossinterpolate(A, tolerance=1e-8)
approx = Matrix(ci)
error = maximum(abs.(A .- approx))
```
"""
function crossinterpolate(
    a::AbstractMatrix{T};
    tolerance=1e-6,
    maxiter=200,
    firstpivot=argmax(abs.(a))
) where {T}
    # 初始化：从第一个枢轴开始
    ci = MatrixCI(a, firstpivot)
    
    # 迭代添加枢轴
    for iter in 1:maxiter
        # 找到误差最大的位置
        pivoterror, newpivot = findmax(localerror(ci, a))
        
        # 检查收敛
        if pivoterror < tolerance
            return ci
        end
        
        # 添加新枢轴
        addpivot!(ci, a, newpivot)
    end
    
    return ci
end
