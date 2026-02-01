# ===================================================================
# abstracttensortrain.jl - 张量列的抽象类型和公共接口
# ===================================================================
# 这个文件定义了所有张量列(Tensor Train)类型的抽象基类和共享函数。
#
# 张量列(TT)，也称为矩阵乘积态(MPS)，是高维张量的低秩表示：
#   f(i₁, i₂, ..., iₙ) = T₁[i₁] * T₂[i₂] * ... * Tₙ[iₙ]
# 其中每个 Tₖ[iₖ] 是一个矩阵。
#
# 这种表示将指数级别的存储 O(d^n) 降低到多项式级别 O(n * d * r²)
# 其中 d 是局部维度，r 是键维度（bond dimension）。
# ===================================================================

# ===================================================================
# 类型别名定义
# ===================================================================
# 为了代码可读性，定义了常用的类型别名

"""
    const LocalIndex = Int

局部索引类型别名。
表示张量列中单个站点(site)的索引值。
"""
const LocalIndex = Int

"""
    const MultiIndex = Vector{LocalIndex}

多重索引类型别名。
表示张量列中所有站点索引的组合，如 [i₁, i₂, ..., iₙ]。
"""
const MultiIndex = Vector{LocalIndex}

# ===================================================================
# 抽象类型定义
# ===================================================================

"""
    abstract type AbstractTensorTrain{V} <: Function end

所有张量列类型的抽象父类型。

# 类型参数
- `V`: 张量元素的值类型（如 Float64, ComplexF64）

# 说明
AbstractTensorTrain继承自Function，这意味着张量列对象可以像函数一样调用：
```julia
tt = TensorTrain(...)
value = tt([1, 2, 3, 4])  # 在索引[1,2,3,4]处求值
```

# 主要实现
- [`TensorTrain`](@ref): 标准张量列
- [`TensorCI2`](@ref): TCI2算法产生的张量列
- [`TensorCI1`](@ref): TCI1算法产生的张量列

# 迭代
张量列可以被迭代，返回其中的每个张量核心：
```julia
for T in tt
    # T 是每个站点的张量
end
```
"""
abstract type AbstractTensorTrain{V} <: Function end

"""
    Base.show(io::IO, tt::AbstractTensorTrain{ValueType}) where {ValueType}

定义张量列的打印格式。

# 输出示例
```
TensorTrain{Float64, 3} with rank 15
```
"""
function Base.show(io::IO, tt::AbstractTensorTrain{ValueType}) where {ValueType}
    print(io, "$(typeof(tt)) with rank $(rank(tt))")
end

# ===================================================================
# 维度相关函数
# ===================================================================

"""
    function linkdims(tt::AbstractTensorTrain{V})::Vector{Int} where {V}

获取张量列中所有连接(link)的维度。

# 背景
张量列表示为: T₁ - T₂ - T₃ - ... - Tₙ
其中 - 表示连接（收缩的指标）

# 返回值
- 长度为 `length(tt)-1` 的整数数组
- `linkdims(tt)[i]` 是第i个和第i+1个张量之间连接的维度

# 数学表示
如果 Tₖ 的形状是 (χₖ₋₁, dₖ, χₖ)，那么 linkdims 返回 [χ₁, χ₂, ..., χₙ₋₁]

# 另见
参见: [`rank`](@ref) 获取最大键维度
"""
function linkdims(tt::AbstractTensorTrain{V})::Vector{Int} where {V}
    # tt[k] 返回第k个张量，size(T, 1) 是其第一个维度（左键维度）
    # 对于第2到最后一个张量，它们的第一个维度就是与前一个张量的连接维度
    return [size(T, 1) for T in tt[2:end]]
end

"""
    function linkdim(tt::AbstractTensorTrain{V}, i::Int)::Int where {V}

获取第i个连接的维度。

# 参数
- `tt`: 张量列
- `i`: 连接索引（1 ≤ i ≤ length(tt)-1）

# 返回值
- 第i个和第i+1个张量之间连接的维度

# 说明
等价于 `size(tt[i+1], 1)`
"""
function linkdim(tt::AbstractTensorTrain{V}, i::Int)::Int where {V}
    return size(sitetensor(tt, i+1), 1)
end

"""
    function sitedims(tt::AbstractTensorTrain{V})::Vector{Vector{Int}} where {V}

获取张量列中所有站点(site)索引的维度。

# 背景
每个张量核心 Tₖ 的形状是 (χₖ₋₁, d₁, d₂, ..., dₘ, χₖ)
其中 d₁, d₂, ..., dₘ 是站点索引（物理指标）的维度

# 返回值
- 向量的向量
- `sitedims(tt)[k]` 是第k个张量的站点维度列表

# 示例
```julia
# 如果 tt[1] 的形状是 (1, 3, 4, 5)
# 那么 sitedims(tt)[1] = [3, 4]
# (第一和最后一个维度是键维度，中间的是站点维度)
```
"""
function sitedims(tt::AbstractTensorTrain{V})::Vector{Vector{Int}} where {V}
    # size(T)[2:end-1] 取中间的维度（去掉第一个和最后一个）
    # collect 将元组转换为数组
    return [collect(size(T)[2:end-1]) for T in tt]
end

"""
    function sitedim(tt::AbstractTensorTrain{V}, i::Int)::Vector{Int} where {V}

获取第i个站点的维度。

# 参数
- `tt`: 张量列
- `i`: 站点索引

# 返回值
- 第i个站点的维度向量
"""
function sitedim(tt::AbstractTensorTrain{V}, i::Int)::Vector{Int} where {V}
    return collect(size(sitetensor(tt, i))[2:end-1])
end

"""
    function rank(tt::AbstractTensorTrain{V}) where {V}

获取张量列的秩（最大键维度）。

# 返回值
- 所有连接维度中的最大值

# 数学意义
秩是衡量张量列表示复杂度的重要指标。
秩越大，能表示的函数越复杂，但存储和计算成本也越高。

# 另见
参见: [`linkdims`](@ref), [`linkdim`](@ref)
"""
function rank(tt::AbstractTensorTrain{V}) where {V}
    return maximum(linkdims(tt))
end

# ===================================================================
# 张量访问函数
# ===================================================================

"""
    sitetensors(tt::AbstractTensorTrain{V}) where {V}

获取张量列中的所有张量核心。

# 返回值
- 张量数组，每个元素是一个多维数组
"""
function sitetensors(tt::AbstractTensorTrain{V}) where {V}
    return tt.sitetensors
end

"""
    sitetensor(tt::AbstractTensorTrain{V}, i) where {V}

获取第i个站点的张量核心。

# 参数
- `tt`: 张量列
- `i`: 站点索引

# 返回值
- 第i个张量核心（多维数组）
"""
function sitetensor(tt::AbstractTensorTrain{V}, i) where {V}
    return tt.sitetensors[i]
end

"""
    function length(tt::AbstractTensorTrain{V}) where {V}

获取张量列的长度（站点数量）。

# 返回值
- 张量核心的数量
"""
function length(tt::AbstractTensorTrain{V}) where {V}
    return length(sitetensors(tt))
end

# ===================================================================
# 迭代器接口
# ===================================================================

"""
    Base.iterate(tt::AbstractTensorTrain{V}) where {V}

实现张量列的迭代起始。

# 说明
使得张量列可以在for循环中使用：
```julia
for T in tt
    println(size(T))
end
```
"""
function Base.iterate(tt::AbstractTensorTrain{V}) where {V}
    return iterate(sitetensors(tt))
end

"""
    Base.iterate(tt::AbstractTensorTrain{V}, state) where {V}

实现张量列的迭代继续。
"""
function Base.iterate(tt::AbstractTensorTrain{V}, state) where {V}
    return iterate(sitetensors(tt), state)
end

"""
    Base.getindex(tt::AbstractTensorTrain{V}, i) where {V}

使用[]语法访问张量核心。

# 语法
```julia
tt[1]     # 第一个张量
tt[end]   # 最后一个张量
tt[2:4]   # 第2到第4个张量
```
"""
function Base.getindex(tt::AbstractTensorTrain{V}, i) where {V}
    return sitetensor(tt, i)
end

"""
    Base.lastindex(tt::AbstractTensorTrain{V}) where {V}

支持 `end` 关键字。

# 说明
使得可以使用 `tt[end]` 语法访问最后一个张量。
"""
function Base.lastindex(tt::AbstractTensorTrain{V}) where {V}
    return lastindex(sitetensors(tt))
end

# ===================================================================
# 求值函数
# ===================================================================

"""
    function evaluate(
        tt::AbstractTensorTrain{V},
        indexset::Union{AbstractVector{LocalIndex}, NTuple{N, LocalIndex}}
    )::V where {N, V}

在指定索引处计算张量列的值。

# 数学公式
```math
f(i_1, i_2, \\ldots, i_n) = T_1[:, i_1, :] \\cdot T_2[:, i_2, :] \\cdot \\ldots \\cdot T_n[:, i_n, :]
```

# 参数
- `tt`: 张量列
- `indexset`: 索引向量或元组，长度必须等于 length(tt)

# 返回值
- 标量值（类型V）

# 示例
```julia
tt = TensorTrain(...)  # 假设长度为4
value = evaluate(tt, [1, 2, 3, 4])  # 计算 f(1,2,3,4)
```

# 效率
对于单次求值，这个方法效率不高（O(n * r²)）。
对于批量求值，建议使用 TTCache。
"""
function evaluate(
    tt::AbstractTensorTrain{V},
    indexset::Union{AbstractVector{LocalIndex},NTuple{N,LocalIndex}}
)::V where {N,V}
    # 检查索引长度是否匹配
    if length(indexset) != length(tt)
        throw(ArgumentError("To evaluate a tt of length $(length(tt)), you have to provide $(length(tt)) indices, but there were $(length(indexset))."))
    end
    
    # 计算矩阵乘积
    # T[:, i, :] 选择第二个维度为i的切片，得到一个矩阵
    # prod 计算所有矩阵的乘积
    # only 从1×1矩阵中提取标量
    return only(prod(T[:, i, :] for (T, i) in zip(tt, indexset)))
end

"""
    function evaluate(tt::AbstractTensorTrain{V}, indexset::CartesianIndex) where {V}

支持 CartesianIndex 类型的索引。

# 参数
- `indexset`: CartesianIndex，如 CartesianIndex(1, 2, 3)
"""
function evaluate(tt::AbstractTensorTrain{V}, indexset::CartesianIndex) where {V}
    return evaluate(tt, Tuple(indexset))
end

"""
    function evaluate(tt, indexset) where multiple indices per site

支持每个站点有多个索引的情况。

# 参数
- `indexset`: 嵌套的向量或元组，每个元素对应一个站点的多个索引

# 示例
```julia
# 如果每个站点有两个索引
tt = TensorTrain(...)  # 张量形状为 (χ, d1, d2, χ)
evaluate(tt, [[1,2], [3,4], [5,6]])  # 计算 f((1,2), (3,4), (5,6))
```
"""
function evaluate(
    tt::AbstractTensorTrain{V},
    indexset::Union{
        AbstractVector{<:AbstractVector{Int}},
        AbstractVector{NTuple{N,Int}},
        NTuple{N,<:AbstractVector{Int}},
        NTuple{N,NTuple{M,Int}}}
) where {N,M,V}
    if length(indexset) != length(tt)
        throw(ArgumentError("To evaluate a tt of length $(length(tt)), you have to provide $(length(tt)) indices, but there were $(length(indexset))."))
    end
    # 检查每个站点的索引数量是否匹配张量的站点维度
    for (n, (T, i)) in enumerate(zip(tt, indexset))
        length(size(T)) == length(i) + 2 || throw(ArgumentError("The index set $(i) at position $n does not have the correct length for the tensor of size $(size(T))."))
    end
    # T[:, i..., :] 展开多个索引
    return only(prod(T[:, i..., :] for (T, i) in zip(tt, indexset)))
end

"""
    (tt::AbstractTensorTrain{V})(indexset) where {V}

使张量列可以像函数一样被调用。

# 语法
```julia
tt([1, 2, 3, 4])  # 等价于 evaluate(tt, [1, 2, 3, 4])
```
"""
function (tt::AbstractTensorTrain{V})(indexset) where {V}
    return evaluate(tt, indexset)
end

# ===================================================================
# 求和函数
# ===================================================================

"""
    function sum(tt::AbstractTensorTrain{V}) where {V}

高效计算张量列在所有格点上的和。

# 数学公式
```math
\\sum_{i_1, i_2, \\ldots, i_n} f(i_1, i_2, \\ldots, i_n)
```

# 算法
不需要遍历所有索引组合（那将是指数级别的）。
利用TT结构，可以逐站点累加：
1. 对每个张量 Tₖ，对站点维度求和得到矩阵
2. 将这些矩阵相乘
复杂度为 O(n * d * r²)，不是 O(d^n)

# 返回值
- 标量值
"""
function sum(tt::AbstractTensorTrain{V}) where {V}
    # sum(tt[1], dims=(1,2)) 对前两个维度求和
    # [1, 1, :] 提取最后一个维度
    # transpose 转置为行向量
    v = transpose(sum(tt[1], dims=(1, 2))[1, 1, :])
    
    # 对后续张量，对站点维度(dim=2)求和
    for T in tt[2:end]
        v *= sum(T, dims=2)[:, 1, :]
    end
    
    # only 从1元素数组中提取标量
    return only(v)
end

# ===================================================================
# 张量列加法
# ===================================================================

"""
    _addtttensor(A, B; factorA=1, factorB=1, lefttensor=false, righttensor=false)

将两个张量核心"堆叠"以实现张量列加法。

# 背景
要计算 C = A + B（其中A, B是张量列），
需要将对应位置的张量核心"堆叠"成块对角结构：
```
C[k] = [factorA*A[k],    0   ]
       [   0,     factorB*B[k]]
```
边界张量特殊处理。

# 参数
- `A`, `B`: 要堆叠的张量
- `factorA`, `factorB`: 乘法因子
- `lefttensor`: 是否是最左边的张量
- `righttensor`: 是否是最右边的张量
"""
function _addtttensor(
    A::Array{V}, B::Array{V};
    factorA=one(V), factorB=one(V),
    lefttensor=false, righttensor=false
) where {V}
    # 检查维度数是否匹配
    if ndims(A) != ndims(B)
        throw(DimensionMismatch("Elementwise addition only works if both tensors have the same indices, but A and B have different numbers ($(ndims(A)) and $(ndims(B))) of indices."))
    end
    
    nd = ndims(A)
    # 确定偏移量（对于边界张量，偏移为0）
    offset1 = lefttensor ? 0 : size(A, 1)
    offset3 = righttensor ? 0 : size(A, nd)
    
    # 中间（站点）索引的切片
    localindices = fill(Colon(), nd - 2)
    
    # 创建输出张量（初始化为0）
    C = zeros(V, offset1 + size(B, 1), size(A)[2:nd-1]..., offset3 + size(B, nd))
    
    # 放置A到左上块
    C[1:size(A, 1), localindices..., 1:size(A, nd)] = factorA * A
    
    # 放置B到右下块
    C[offset1+1:end, localindices..., offset3+1:end] = factorB * B
    
    return C
end

@doc raw"""
    function add(
        lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V};
        factorlhs=one(V), factorrhs=one(V),
        tolerance::Float64=0.0, maxbonddim::Int=typemax(Int)
    ) where {V}

两个张量列相加。

# 数学描述
计算 `C(v) = factorlhs * lhs(v) + factorrhs * rhs(v)`

# 参数
- `lhs`, `rhs`: 要相加的张量列
- `factorlhs`, `factorrhs`: 乘法因子
- `tolerance`, `maxbonddim`: 重压缩参数

# 返回值
- 新的 TensorTrain，表示和

# 注意
加法会增加键维度：χ_result = χ_lhs + χ_rhs
可以使用 tolerance 和 maxbonddim 进行后续压缩。

# 另见
参见: [`+`](@ref)
"""
function add(
    lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V};
    factorlhs=one(V), factorrhs=one(V),
    tolerance::Float64=0.0, maxbonddim::Int=typemax(Int)
) where {V}
    # 检查长度是否匹配
    if length(lhs) != length(rhs)
        throw(DimensionMismatch("Two tensor trains with different length ($(length(lhs)) and $(length(rhs))) cannot be added elementwise."))
    end
    
    L = length(lhs)
    
    # 构造加法后的张量列
    tt = tensortrain(
        [
            _addtttensor(
                lhs[ell], rhs[ell];
                factorA=((ell == L) ? factorlhs : one(V)),  # 因子只在最后一个张量应用
                factorB=((ell == L) ? factorrhs : one(V)),
                lefttensor=(ell==1),     # 标记边界
                righttensor=(ell==L)
            )
            for ell in 1:L
        ]
    )
    
    # 进行压缩
    compress!(tt, :SVD; tolerance, maxbonddim)
    return tt
end

@doc raw"""
    function subtract(
        lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V};
        tolerance::Float64=0.0, maxbonddim::Int=typemax(Int)
    )

两个张量列相减。

# 数学描述
计算 `C(v) = lhs(v) - rhs(v)`

# 实现
调用 add(lhs, rhs; factorrhs=-1, ...)
"""
function subtract(
    lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V};
    tolerance::Float64=0.0, maxbonddim::Int=typemax(Int)
) where {V}
    return add(lhs, rhs; factorrhs=-1 * one(V), tolerance, maxbonddim)
end

@doc raw"""
    function (+)(lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V}) where {V}

张量列加法运算符。

# 语法
```julia
C = A + B
```

# 注意
结果的键维度是两个输入键维度之和。
如需压缩，使用 [`add`](@ref) 函数。
"""
function Base.:+(lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V}) where {V}
    return add(lhs, rhs)
end

@doc raw"""
    function (-)(lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V}) where {V}

张量列减法运算符。

# 语法
```julia
C = A - B
```
"""
function Base.:-(lhs::AbstractTensorTrain{V}, rhs::AbstractTensorTrain{V}) where {V}
    return subtract(lhs, rhs)
end

# ===================================================================
# 范数计算
# ===================================================================

"""
    LA.norm2(tt::AbstractTensorTrain{V})::Float64 where {V}

计算张量列的Frobenius范数的平方。

# 数学公式
```math
\\|TT\\|_F^2 = \\sum_{i_1, \\ldots, i_n} |f(i_1, \\ldots, i_n)|^2
```

# 算法
使用张量网络收缩高效计算，不需要显式枚举所有索引。
"""
function LA.norm2(tt::AbstractTensorTrain{V})::Float64 where {V}
    # 内部函数：计算单个站点的贡献
    function _f(n)::Matrix{V}
        t = sitetensor(tt, n)
        # 将多个站点索引合并为一个
        t3 = reshape(t, size(t)[1], :, size(t)[end])
        # 收缩站点索引：(lc, s, rc) * (l, s, r) => (lc, rc, l, r)
        tct = _contract(conj.(t3), t3, (2,), (2,))
        # 重排为 (lc, l, rc, r) 并合并为矩阵
        tct = permutedims(tct, (1, 3, 2, 4))
        return reshape(tct, size(tct, 1) * size(tct, 2), size(tct, 3) * size(tct, 4))
    end
    # 将所有站点的贡献矩阵相乘
    return real(only(reduce(*, (_f(n) for n in 1:length(tt)))))
end

"""
    LA.norm(tt::AbstractTensorTrain{V})::Float64 where {V}

计算张量列的Frobenius范数。

# 公式
```math
\\|TT\\|_F = \\sqrt{\\sum_{i_1, \\ldots, i_n} |f(i_1, \\ldots, i_n)|^2}
```
"""
function LA.norm(tt::AbstractTensorTrain{V})::Float64 where {V}
    sqrt(LA.norm2(tt))
end