# ===================================================================
# matrixluci.jl - 基于LU分解的矩阵交叉插值
# ===================================================================
# 这个文件实现了MatrixLUCI类型，它是基于秩揭示LU分解的矩阵交叉插值。
#
# LU分解将矩阵分解为下三角矩阵L和上三角矩阵U的乘积：A = L * U
# 秩揭示LU分解还能确定矩阵的有效秩，并选择最重要的行和列。
#
# MatrixLUCI是rrLU的包装器，提供与AbstractMatrixCI一致的接口。
# ===================================================================

"""
    mutable struct MatrixLUCI{T} <: AbstractMatrixCI{T}

基于秩揭示LU分解的矩阵交叉插值。

# 类型参数
- `T`: 矩阵元素类型

# 字段
- `lu::rrLU{T}`: 底层的秩揭示LU分解对象

# 描述
MatrixLUCI将rrLU分解包装成符合AbstractMatrixCI接口的对象，
使得可以用统一的方式处理不同的矩阵交叉插值算法。

# 优点
- 利用LU分解的数值稳定性
- 自动进行秩揭示，确定有效秩
- 支持容差控制，可以截断小枢轴
"""
mutable struct MatrixLUCI{T} <: AbstractMatrixCI{T}
    lu::rrLU{T}  # 内部存储的rrLU分解对象
end

"""
    MatrixLUCI(A::AbstractMatrix{T}; kwargs...) where {T}

从矩阵直接创建MatrixLUCI。

# 参数
- `A::AbstractMatrix{T}`: 要分解的矩阵
- `kwargs...`: 传递给rrlu的关键字参数，如：
  - `maxrank`: 最大秩
  - `reltol`: 相对容差
  - `abstol`: 绝对容差
  - `leftorthogonal`: 是否左正交化

# 示例
```julia
A = rand(100, 50)
luci = MatrixLUCI(A, reltol=1e-10)
```
"""
function MatrixLUCI(A::AbstractMatrix{T}; kwargs...) where {T}
    MatrixLUCI{T}(rrlu(A; kwargs...))
end

"""
    MatrixLUCI(::Type{ValueType}, f, matrixsize, I0, J0; kwargs...) where {ValueType}

使用函数求值器创建MatrixLUCI（不需要存储完整矩阵）。

# 参数
- `ValueType`: 元素类型
- `f`: 矩阵元素求值函数 f(i, j) 返回位置(i,j)的元素
- `matrixsize::Tuple{Int,Int}`: 矩阵尺寸 (nrows, ncols)
- `I0::AbstractVector{Int}`: 初始行索引集
- `J0::AbstractVector{Int}`: 初始列索引集
- `kwargs...`: 其他参数

# 用途
当矩阵太大无法完全存储时，可以用函数来按需计算矩阵元素。
"""
function MatrixLUCI(
    ::Type{ValueType},
    f,
    matrixsize::Tuple{Int,Int},
    I0::AbstractVector{Int}=Int[],
    J0::AbstractVector{Int}=Int[];
    kwargs...
) where {ValueType}
    MatrixLUCI{ValueType}(rrlu(ValueType, f, matrixsize, I0, J0; kwargs...))
end

"""
    size(luci::MatrixLUCI{T}) where {T}

返回矩阵尺寸。

# 返回值
- `(nrows, ncols)` 元组
"""
function size(luci::MatrixLUCI{T}) where {T}
    return size(luci.lu)
end

"""
    size(luci::MatrixLUCI{T}, dim) where {T}

返回矩阵在指定维度的尺寸。

# 参数
- `dim`: 1 表示行数，2 表示列数

# 返回值
- 指定维度的尺寸
"""
function size(luci::MatrixLUCI{T}, dim) where {T}
    return size(luci.lu, dim)
end

"""
    npivots(luci::MatrixLUCI{T}) where {T}

返回枢轴数量（即近似的秩）。

# 返回值
- 枢轴/秩的数量
"""
function npivots(luci::MatrixLUCI{T}) where {T}
    return npivots(luci.lu)
end

"""
    rowindices(luci::MatrixLUCI{T}) where {T}

返回选中的枢轴行索引。

# 返回值
- 长度为npivots的整数数组
"""
function rowindices(luci::MatrixLUCI{T}) where {T}
    return rowindices(luci.lu)
end

"""
    colindices(luci::MatrixLUCI{T}) where {T}

返回选中的枢轴列索引。

# 返回值
- 长度为npivots的整数数组
"""
function colindices(luci::MatrixLUCI{T}) where {T}
    return colindices(luci.lu)
end

"""
    colmatrix(luci::MatrixLUCI{T}) where {T}

返回枢轴列组成的矩阵 A[:, J]。

# 返回值
- `(nrows × npivots)` 矩阵

# 说明
与rowmatrix一起，这两个矩阵可以重构原矩阵的近似：
A ≈ colmatrix(luci) * (pivotmatrix)^(-1) * rowmatrix(luci)
"""
function colmatrix(luci::MatrixLUCI{T}) where {T}
    # left(lu) 是L矩阵
    # right(lu, permute=false) 是未排列的U矩阵
    return left(luci.lu) * right(luci.lu, permute=false)[:, 1:npivots(luci)]
end

"""
    rowmatrix(luci::MatrixLUCI{T}) where {T}

返回枢轴行组成的矩阵 A[I, :]。

# 返回值
- `(npivots × ncols)` 矩阵
"""
function rowmatrix(luci::MatrixLUCI{T}) where {T}
    return left(luci.lu, permute=false)[1:npivots(luci), :] * right(luci.lu)
end

"""
    colstimespivotinv(luci::MatrixLUCI{T}) where {T}

计算 C * P^(-1)，其中 C 是列矩阵，P 是枢轴矩阵。

# 返回值
- `(nrows × npivots)` 矩阵

# 数学说明
这是交叉插值分解的左因子：A ≈ (C * P^(-1)) * R
其中R是行矩阵。

# 实现细节
使用LU分解的性质来高效计算，避免显式求逆。
"""
function colstimespivotinv(luci::MatrixLUCI{T}) where {T}
    n = npivots(luci)
    
    # 创建单位矩阵作为基础
    # Matrix{T}(I, m, n) 创建 m×n 的单位矩阵（I是LinearAlgebra中的单位算子）
    result = Matrix{T}(I, size(luci, 1), n)
    
    # 如果枢轴数小于行数，需要计算额外的行
    if n < size(luci, 1)
        L = left(luci.lu; permute=false)  # 获取未排列的L矩阵
        # 使用三角求解：L[n+1:end, :] / L[1:n, :]
        # 这等价于 L[n+1:end, :] * inv(L[1:n, :])
        # LowerTriangular告诉Julia这是下三角矩阵，可以更高效地求解
        result[n+1:end, :] = L[n+1:end, :] / LowerTriangular(L[1:n, :])
    end
    
    # 应用行排列：将排列后的结果放回原始行顺序
    result[luci.lu.rowpermutation, :] = result
    return result
end

"""
    pivotinvtimesrows(luci::MatrixLUCI{T}) where {T}

计算 P^(-1) * R，其中 P 是枢轴矩阵，R 是行矩阵。

# 返回值
- `(npivots × ncols)` 矩阵

# 数学说明
这是交叉插值分解的右因子：A ≈ C * (P^(-1) * R)
其中C是列矩阵。
"""
function pivotinvtimesrows(luci::MatrixLUCI{T}) where {T}
    n = npivots(luci)
    result = Matrix{T}(I, n, size(luci, 2))
    
    if n < size(luci, 2)
        U = right(luci.lu; permute=false)
        # 使用三角求解：inv(U[:, 1:n]) * U[:, n+1:end]
        # UpperTriangular告诉Julia这是上三角矩阵
        result[:, n+1:end] = UpperTriangular(U[:, 1:n]) \ U[:, n+1:end]
    end
    
    # 应用列排列
    result[:, luci.lu.colpermutation] = result
    return result
end

"""
    left(luci::MatrixLUCI{T}) where {T}

获取分解的左因子。

# 返回值
根据leftorthogonal设置返回不同的矩阵：
- 如果左正交：返回 colstimespivotinv（使左因子的列正交）
- 否则：返回 colmatrix

# 用途
用于重构：A ≈ left(luci) * right(luci)
"""
function left(luci::MatrixLUCI{T}) where {T}
    if luci.lu.leftorthogonal
        return colstimespivotinv(luci)
    else
        return colmatrix(luci)
    end
end

"""
    right(luci::MatrixLUCI{T}) where {T}

获取分解的右因子。

# 返回值
根据leftorthogonal设置返回不同的矩阵：
- 如果左正交：返回 rowmatrix
- 否则：返回 pivotinvtimesrows

# 用途
用于重构：A ≈ left(luci) * right(luci)
"""
function right(luci::MatrixLUCI{T}) where {T}
    if luci.lu.leftorthogonal
        return rowmatrix(luci)
    else
        return pivotinvtimesrows(luci)
    end
end

"""
    pivoterrors(luci::MatrixLUCI{T}) where {T}

返回各枢轴的误差估计。

# 返回值
- 误差数组，长度为 npivots+1
- 最后一个元素是添加当前枢轴后的残差估计
"""
function pivoterrors(luci::MatrixLUCI{T}) where {T}
    return pivoterrors(luci.lu)
end

"""
    lastpivoterror(luci::MatrixLUCI{T}) where {T}

返回最后一个枢轴的误差。

# 返回值
- 最后添加的枢轴的误差估计

# 用途
用于判断是否需要继续添加枢轴（如果误差足够小，可以停止）。
"""
function lastpivoterror(luci::MatrixLUCI{T}) where {T}
    return lastpivoterror(luci.lu)
end
