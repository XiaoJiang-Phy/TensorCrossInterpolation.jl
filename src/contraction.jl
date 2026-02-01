# ===================================================================
# contraction.jl - 张量列收缩
# ===================================================================
# 这个文件实现了两个张量列算子（TTO/MPO）的收缩操作。
#
# 张量列收缩是量子多体物理和机器学习中的核心操作。
# 给定两个MPO A 和 B，计算它们的乘积 C = A * B。
#
# 提供三种收缩算法：
#   1. :TCI - 使用 TCI2 算法拟合收缩结果
#   2. :naive - 朴素收缩后 SVD 压缩
#   3. :zipup - 边收缩边 LU/SVD 分解
#
# 参考文献：
# - https://tensornetwork.org/mps/algorithms/zip_up_mpo/
# ===================================================================

"""
    struct Contraction{T} <: BatchEvaluator{T}

两个张量列算子（TTO/MPO）的收缩对象。

# 背景
在量子多体物理中，我们经常需要计算两个算子的乘积。
对于矩阵乘积算子（MPO），这可以表示为：
```
C_{i,k} = ∑_j A_{i,j} * B_{j,k}
```

# 类型参数
- `T`: 张量元素类型

# 字段
- `mpo::NTuple{2,TensorTrain{T,4}}`: 两个4腿张量列（MPO）
- `leftcache::Dict`: 左环境缓存
- `rightcache::Dict`: 右环境缓存
- `f::Union{Nothing,Function}`: 可选的后处理函数
- `sitedims::Vector{Vector{Int}}`: 站点维度（融合后）

# 张量索引约定
每个站点张量是4维的：
- 第1维：左键维度
- 第2维：上物理指标（A的第一个自由指标）
- 第3维：下物理指标（与B收缩的指标）
- 第4维：右键维度

# 创建方式
```julia
A = TensorTrain{Float64,4}(...)  # 4腿张量列
B = TensorTrain{Float64,4}(...)
contraction = Contraction(A, B)
result = contraction([1, 2, 3])  # 求值
```

# 用途
- MPO-MPO 乘积
- 量子态的时间演化算子应用
- 算子的连续作用
"""
struct Contraction{T} <: BatchEvaluator{T}
    mpo::NTuple{2,TensorTrain{T,4}}                  # 两个 MPO
    leftcache::Dict{Vector{Tuple{Int,Int}},Matrix{T}}   # 左环境缓存
    rightcache::Dict{Vector{Tuple{Int,Int}},Matrix{T}}  # 右环境缓存
    f::Union{Nothing,Function}                        # 后处理函数
    sitedims::Vector{Vector{Int}}                    # 站点维度
end

"""
    Base.length(obj::Contraction)

获取收缩对象的长度（站点数量）。
"""
Base.length(obj::Contraction) = length(obj.mpo[1])

"""
    sitedims(obj::Contraction{T}) where {T}

获取收缩结果的站点维度。
"""
function sitedims(obj::Contraction{T})::Vector{Vector{Int}} where {T}
    return obj.sitedims
end

function Base.lastindex(obj::Contraction{T}) where {T}
    return lastindex(obj.mpo[1])
end

function Base.getindex(obj::Contraction{T}, i) where {T}
    return getindex(obj.mpo[1], i)
end

function Base.show(io::IO, obj::Contraction{T}) where {T}
    print(
        io,
        "$(typeof(obj)) of tensor trains with ranks $(rank(obj.mpo[1])) and $(rank(obj.mpo[2]))",
    )
end

"""
    Contraction(a::TensorTrain{T,4}, b::TensorTrain{T,4}; f=nothing) where {T}

创建两个MPO的收缩对象。

# 参数
- `a, b`: 要收缩的两个MPO
- `f`: 可选的后处理函数（应用于每个元素）

# 要求
- `a` 和 `b` 必须具有相同的长度
- `a` 的下指标必须与 `b` 的上指标匹配

# 示例
```julia
A = TensorTrain{Float64,4}(...)
B = TensorTrain{Float64,4}(...)
C = Contraction(A, B)
# C 可以像函数一样调用
value = C([1, 2, 3])
```
"""
function Contraction(
    a::TensorTrain{T,4},
    b::TensorTrain{T,4};
    f::Union{Nothing,Function}=nothing,
) where {T}
    mpo = a, b
    
    # 验证长度匹配
    if length(unique(length.(mpo))) > 1
        throw(ArgumentError("Tensor trains must have the same length."))
    end
    
    # 验证收缩指标匹配
    for n = 1:length(mpo[1])
        if size(mpo[1][n], 3) != size(mpo[2][n], 2)
            error("Tensor trains must share the identical index at n=$(n)!")
        end
    end

    # 计算结果的站点维度
    localdims1 = [size(mpo[1][n], 2) for n = 1:length(mpo[1])]
    localdims3 = [size(mpo[2][n], 3) for n = 1:length(mpo[2])]
    sitedims = [[x, y] for (x, y) in zip(localdims1, localdims3)]

    return Contraction(
        mpo,
        Dict{Vector{Tuple{Int,Int}},Matrix{T}}(),
        Dict{Vector{Tuple{Int,Int}},Matrix{T}}(),
        f,
        sitedims
    )
end

# ===================================================================
# 辅助函数
# ===================================================================

"""
    _localdims(obj, n)

获取位置 n 的局部维度。
"""
_localdims(obj::TensorTrain{<:Any,4}, n::Int)::Tuple{Int,Int} =
    (size(obj[n], 2), size(obj[n], 3))
_localdims(obj::Contraction{<:Any}, n::Int)::Tuple{Int,Int} =
    (size(obj.mpo[1][n], 2), size(obj.mpo[2][n], 3))

"""
    _getindex(x, indices)

从元组中提取指定索引的元素。
"""
_getindex(x, indices) = ntuple(i -> x[indices[i]], length(indices))

"""
    _contract(a, b, idx_a, idx_b)

通用张量收缩函数。

# 参数
- `a, b`: 要收缩的张量
- `idx_a, idx_b`: 要收缩的指标

# 返回值
收缩后的张量

# 算法
1. 将目标指标重排到最后（a）或最前（b）
2. 重塑为矩阵
3. 矩阵乘法
4. 重塑回张量
"""
function _contract(
    a::AbstractArray{T1,N1},
    b::AbstractArray{T2,N2},
    idx_a::NTuple{n1,Int},
    idx_b::NTuple{n2,Int}
) where {T1,T2,N1,N2,n1,n2}
    # 验证输入
    length(idx_a) == length(idx_b) || error("length(idx_a) != length(idx_b)")
    length(unique(idx_a)) == length(idx_a) || error("idx_a contains duplicate elements")
    length(unique(idx_b)) == length(idx_b) || error("idx_b contains duplicate elements")
    all(1 <= idx <= N1 for idx in idx_a) || error("idx_a contains elements out of range")
    all(1 <= idx <= N2 for idx in idx_b) || error("idx_b contains elements out of range")

    # 找出不参与收缩的指标
    rest_idx_a = setdiff(1:N1, idx_a)
    rest_idx_b = setdiff(1:N2, idx_b)

    # 重塑并收缩
    amat = reshape(permutedims(a, (rest_idx_a..., idx_a...)), prod(_getindex(size(a), rest_idx_a)), prod(_getindex(size(a), idx_a)))
    bmat = reshape(permutedims(b, (idx_b..., rest_idx_b...)), prod(_getindex(size(b), idx_b)), prod(_getindex(size(b), rest_idx_b)))

    return reshape(amat * bmat, _getindex(size(a), rest_idx_a)..., _getindex(size(b), rest_idx_b)...)
end

"""
    _unfuse_idx(obj::Contraction{T}, n::Int, idx::Int)

将融合索引转换为分离索引 (i, j)。

# 说明
收缩结果有融合的站点维度 d1 * d2。
这个函数将线性索引转换回 (i, j) 形式。
"""
function _unfuse_idx(obj::Contraction{T}, n::Int, idx::Int)::Tuple{Int,Int} where {T}
    return reverse(divrem(idx - 1, _localdims(obj, n)[1]) .+ 1)
end

"""
    _fuse_idx(obj::Contraction{T}, n::Int, idx::Tuple{Int,Int})

将分离索引 (i, j) 转换为融合索引。
"""
function _fuse_idx(obj::Contraction{T}, n::Int, idx::Tuple{Int,Int})::Int where {T}
    return idx[1] + _localdims(obj, n)[1] * (idx[2] - 1)
end

"""
    _extend_cache(oldcache::Matrix{T}, a_ell, b_ell, i, j) where {T}

扩展左/右环境缓存。

# 参数
- `oldcache`: 上一层的环境
- `a_ell, b_ell`: 当前层的张量
- `i, j`: 当前层的物理指标

# 算法
1. 收缩左键：oldcache * a
2. 收缩共享指标：result * b
"""
function _extend_cache(oldcache::Matrix{T}, a_ell::Array{T,4}, b_ell::Array{T,4}, i::Int, j::Int) where {T}
    # (link_a, link_b) * (link_a, s, link_a') => (link_b, s, link_a')
    tmp1 = _contract(oldcache, a_ell[:, i, :, :], (1,), (1,))

    # (link_b, s, link_a') * (link_b, s, link_b') => (link_a', link_b')
    return _contract(tmp1, b_ell[:, :, j, :], (1, 2), (1, 2))
end

# ===================================================================
# 环境求值（带缓存）
# ===================================================================

"""
    evaluateleft(obj::Contraction{T}, indexset) where {T}

计算左环境。

# 参数
- `indexset`: Tuple{Int,Int} 的向量，表示物理指标

# 返回值
- 矩阵 (link_a, link_b)

# 缓存
使用 `leftcache` 字典缓存已计算的结果。
"""
function evaluateleft(
    obj::Contraction{T},
    indexset::AbstractVector{Tuple{Int,Int}},
)::Matrix{T} where {T}
    if length(indexset) >= length(obj.mpo[1])
        error("Invalid indexset: $indexset")
    end

    a, b = obj.mpo

    # 空索引集返回单位矩阵
    if length(indexset) == 0
        return ones(T, 1, 1)
    end

    ell = length(indexset)
    
    # 第一个位置的特殊处理
    if ell == 1
        i, j = indexset[1]
        return transpose(a[1][1, i, :, :]) * b[1][1, :, j, :]
    end

    # 检查缓存
    key = collect(indexset)
    if !(key in keys(obj.leftcache))
        i, j = indexset[end]
        obj.leftcache[key] = _extend_cache(evaluateleft(obj, indexset[1:ell-1]), a[ell], b[ell], i, j)
    end

    return obj.leftcache[key]
end

"""
    evaluateright(obj::Contraction{T}, indexset) where {T}

计算右环境。

# 参数
- `indexset`: Tuple{Int,Int} 的向量，表示物理指标

# 返回值
- 矩阵 (link_a, link_b)
"""
function evaluateright(
    obj::Contraction{T},
    indexset::AbstractVector{Tuple{Int,Int}},
)::Matrix{T} where {T}
    if length(indexset) >= length(obj.mpo[1])
        error("Invalid indexset: $indexset")
    end

    a, b = obj.mpo
    N = length(obj)

    if length(indexset) == 0
        return ones(T, 1, 1)
    elseif length(indexset) == 1
        i, j = indexset[1]
        return a[end][:, i, :, 1] * transpose(b[end][:, :, j, 1])
    end

    ell = N - length(indexset) + 1

    key = collect(indexset)
    if !(key in keys(obj.rightcache))
        i, j = indexset[1]
        obj.rightcache[key] = _extend_cache(
            evaluateright(obj, indexset[2:end]),
            permutedims(a[ell], (4, 2, 3, 1)),
            permutedims(b[ell], (4, 2, 3, 1)),
            i, j)
    end

    return obj.rightcache[key]
end

# ===================================================================
# 求值函数
# ===================================================================

"""
    evaluate(obj::Contraction{T}, indexset::AbstractVector{Int}) where {T}

使用融合索引求值收缩。
"""
function evaluate(obj::Contraction{T}, indexset::AbstractVector{Int})::T where {T}
    if length(obj) != length(indexset)
        error("Length mismatch: $(length(obj)) != $(length(indexset))")
    end

    indexset_unfused = [_unfuse_idx(obj, n, indexset[n]) for n = 1:length(obj)]
    return evaluate(obj, indexset_unfused)
end

"""
    evaluate(obj::Contraction{T}, indexset::AbstractVector{Tuple{Int,Int}}) where {T}

使用分离索引求值收缩。

# 算法
1. 从中间将索引分为左右两部分
2. 计算左环境和右环境
3. 收缩得到结果
4. 可选：应用后处理函数
"""
function evaluate(
    obj::Contraction{T},
    indexset::AbstractVector{Tuple{Int,Int}},
)::T where {T}
    if length(obj) != length(indexset)
        error("Length mismatch: $(length(obj)) != $(length(indexset))")
    end

    # 从中间分割
    midpoint = div(length(obj), 2)
    res = sum(
        evaluateleft(obj, indexset[1:midpoint]) .*
        evaluateright(obj, indexset[midpoint+1:end]),
    )

    # 应用后处理函数
    if obj.f isa Function
        return obj.f(res)
    else
        return res
    end
end

"""
    _lineari(dims, mi)

将多索引转换为线性索引。
"""
function _lineari(dims, mi)::Integer
    ci = CartesianIndex(Tuple(mi))
    li = LinearIndices(Tuple(dims))
    return li[ci]
end

"""
    lineari(sitedims, indexset)

将多索引向量转换为线性索引向量。
"""
function lineari(sitedims::Vector{Vector{Int}}, indexset::Vector{MultiIndex})::Vector{Int}
    return [_lineari(sitedims[l], indexset[l]) for l in 1:length(indexset)]
end

# ===================================================================
# 可调用接口
# ===================================================================

"""
    (obj::Contraction{T})(indexset::AbstractVector{Int}) where {T}

使Contraction可调用（融合索引版本）。
"""
function (obj::Contraction{T})(indexset::AbstractVector{Int})::T where {T}
    return evaluate(obj, indexset)
end

"""
    (obj::Contraction{T})(indexset::AbstractVector{<:AbstractVector{Int}}) where {T}

使Contraction可调用（多索引版本）。
"""
function (obj::Contraction{T})(indexset::AbstractVector{<:AbstractVector{Int}})::T where {T}
    return evaluate(obj, lineari(obj.sitedims, indexset))
end

"""
    (obj::Contraction{T})(leftindexset, rightindexset, ::Val{M}) where {T,M}

批量求值接口。
"""
function (obj::Contraction{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    return batchevaluate(obj, leftindexset, rightindexset, Val(M))
end

"""
    batchevaluate(obj::Contraction{T}, leftindexset, rightindexset, ::Val{M}; projector=nothing) where {T,M}

批量求值收缩。

# 参数
- `leftindexset`: 左索引集
- `rightindexset`: 右索引集
- `Val{M}`: 中间自由指标数量
- `projector`: 可选的投影器

# 返回值
形状为 (|left|, site_dims..., |right|) 的数组

# 算法
1. 计算所有左环境
2. 计算所有右环境
3. 逐站点收缩中间张量
4. 与右环境收缩得到最终结果
"""
function batchevaluate(obj::Contraction{T},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
    projector::Union{Nothing,AbstractVector{<:AbstractVector{<:Integer}}}=nothing)::Array{T,M + 2} where {T,M}
    N = length(obj)
    Nr = length(rightindexset[1])
    s_ = length(leftindexset[1]) + 1
    e_ = N - length(rightindexset[1])
    a, b = obj.mpo

    # 处理投影器
    if projector === nothing
        projector = [fill(0, length(obj.sitedims[n])) for n in s_:e_]
    end
    length(projector) == M || error("Length mismatch: length of projector (=$(length(projector))) must be $(M)")
    for n in s_:e_
        length(projector[n - s_ + 1]) == 2 || error("Invalid projector at $n: $(projector[n - s_ + 1]), the length must be 2")
        all(0 .<= projector[n - s_ + 1] .<= obj.sitedims[n]) || error("Invalid projector: $(projector[n - s_ + 1])")
    end

    # 解融合索引
    leftindexset_unfused = [
        [_unfuse_idx(obj, n, idx) for (n, idx) in enumerate(idxs)] for idxs in leftindexset
    ]
    rightindexset_unfused = [
        [_unfuse_idx(obj, N - Nr + n, idx) for (n, idx) in enumerate(idxs)] for
        idxs in rightindexset
    ]

    t1 = time_ns()
    linkdims_a = vcat(1, linkdims(a), 1)
    linkdims_b = vcat(1, linkdims(b), 1)

    # 计算所有左环境
    left_ = Array{T,3}(undef, length(leftindexset), linkdims_a[s_], linkdims_b[s_])
    for (i, idx) in enumerate(leftindexset_unfused)
        left_[i, :, :] .= evaluateleft(obj, idx)
    end
    t2 = time_ns()

    # 计算所有右环境
    right_ = Array{T,3}(
        undef,
        linkdims_a[e_+1],
        linkdims_b[e_+1],
        length(rightindexset),
    )
    for (i, idx) in enumerate(rightindexset_unfused)
        right_[:, :, i] .= evaluateright(obj, idx)
    end
    t3 = time_ns()

    # 逐站点收缩
    leftobj::Array{T,4} = reshape(left_, size(left_)..., 1)
    return_size_siteinds = Int[]
    for n = s_:e_
        slice_ab, shape_ab = projector_to_slice(projector[n - s_ + 1])
        a_n = begin
            a_n_org = obj.mpo[1][n]
            tmp = a_n_org[:, slice_ab[1], :, :]
            reshape(tmp, size(a_n_org, 1), shape_ab[1], size(a_n_org)[3:4]...)
        end
        b_n = begin
            b_n_org = obj.mpo[2][n]
            tmp = b_n_org[:, :, slice_ab[2], :]
            reshape(tmp, size(b_n_org, 1), size(b_n_org, 2), shape_ab[2], size(b_n_org, 4))
        end
        push!(return_size_siteinds, size(a_n, 2) * size(b_n, 3))

        # 收缩步骤
        tmp1 = _contract(leftobj, a_n, (2,), (1,))
        tmp2 = _contract(tmp1, b_n, (2, 5), (1, 2))
        tmp3 = permutedims(tmp2, (1, 4, 6, 2, 3, 5))
        leftobj = reshape(tmp3, size(tmp3)[1:3]..., :)
    end

    return_size = (
        length(leftindexset),
        return_size_siteinds...,
        length(rightindexset),
    )
    t5 = time_ns()

    # 最终收缩
    res = _contract(leftobj, right_, (2, 3), (1, 2))

    if obj.f isa Function
        res .= obj.f.(res)
    end

    return reshape(res, return_size)
end

# ===================================================================
# 收缩算法实现
# ===================================================================

"""
    _contractsitetensors(a::Array{T,4}, b::Array{T,4}) where {T}

收缩两个站点张量。

# 索引约定
- a: (link_a, s1, s2, link_a')
- b: (link_b, s2, s3, link_b')
- 结果: (link_a*link_b, s1, s3, link_a'*link_b')
"""
function _contractsitetensors(a::Array{T,4}, b::Array{T,4})::Array{T,4} where {T}
    # 收缩共享指标
    ab::Array{T,6} = _contract(a, b, (3,), (2,))
    # 重排指标
    abpermuted = permutedims(ab, (1, 4, 2, 5, 3, 6))
    # 融合键维度
    return reshape(abpermuted,
        size(a, 1) * size(b, 1),
        size(a, 2), size(b, 3),
        size(a, 4) * size(b, 4)
    )
end

"""
    contract_naive(a::TensorTrain{T,4}, b::TensorTrain{T,4}; tolerance, maxbonddim) where {T}

朴素收缩算法。

# 算法
1. 逐站点收缩张量
2. 使用 SVD 压缩结果

# 复杂度
键维度可能增长到 χ_A * χ_B
"""
function contract_naive(
    a::TensorTrain{T,4}, b::TensorTrain{T,4};
    tolerance=0.0, maxbonddim=typemax(Int)
)::TensorTrain{T,4} where {T}
    return contract_naive(Contraction(a, b); tolerance, maxbonddim)
end

function contract_naive(
    obj::Contraction{T};
    tolerance=0.0, maxbonddim=typemax(Int)
)::TensorTrain{T,4} where {T}
    if obj.f isa Function
        error("Naive contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
    end

    a, b = obj.mpo
    tt = TensorTrain{T,4}(_contractsitetensors.(sitetensors(a), sitetensors(b)))
    if tolerance > 0 || maxbonddim < typemax(Int)
        compress!(tt, :SVD; tolerance, maxbonddim)
    end
    return tt
end

"""
    _reshape_fusesites(t::AbstractArray{T}) where {T}

融合站点维度（用于 TCI 收缩）。
"""
function _reshape_fusesites(t::AbstractArray{T}) where {T}
    shape = size(t)
    return reshape(t, shape[1], prod(shape[2:end-1]), shape[end]), shape[2:end-1]
end

"""
    _reshape_splitsites(t::AbstractArray{T}, legdims) where {T}

分离站点维度。
"""
function _reshape_splitsites(
    t::AbstractArray{T},
    legdims::Union{AbstractVector{Int},Tuple},
) where {T}
    return reshape(t, size(t, 1), legdims..., size(t, ndims(t)))
end

"""
    _findinitialpivots(f, localdims, nmaxpivots)

为 TCI 收缩寻找初始枢轴。
"""
function _findinitialpivots(f, localdims, nmaxpivots)::Vector{MultiIndex}
    pivots = MultiIndex[]
    for _ in 1:nmaxpivots
        pivot_ = [rand(1:d) for d in localdims]
        pivot_ = optfirstpivot(f, localdims, pivot_)
        if abs(f(pivot_)) == 0.0
            continue
        end
        push!(pivots, pivot_)
    end
    return pivots
end

"""
    contract_TCI(A::TensorTrain{ValueType,4}, B::TensorTrain{ValueType,4}; kwargs...) where {ValueType}

使用 TCI 算法收缩两个 MPO。

# 算法
将收缩视为一个函数，使用 crossinterpolate2 拟合。

# 优势
- 可以直接控制输出的键维度
- 可以应用后处理函数
- 对于低秩结果更高效
"""
function contract_TCI(
    A::TensorTrain{ValueType,4},
    B::TensorTrain{ValueType,4};
    initialpivots::Union{Int,Vector{MultiIndex}}=10,
    f::Union{Nothing,Function}=nothing,
    kwargs...
) where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end
    if !all([sitedim(A, i)[2] == sitedim(B, i)[1] for i = 1:length(A)])
        throw(
            ArgumentError(
                "Cannot contract tensor trains with non-matching site dimensions.",
            ),
        )
    end
    
    matrixproduct = Contraction(A, B; f=f)
    localdims = prod.(matrixproduct.sitedims)
    
    if initialpivots isa Int
        initialpivots = _findinitialpivots(matrixproduct, localdims, initialpivots)
        if isempty(initialpivots)
            error("No initial pivots found.")
        end
    end

    tci, ranks, errors = crossinterpolate2(
        ValueType,
        matrixproduct,
        localdims,
        initialpivots;
        kwargs...,
    )
    
    legdims = [_localdims(matrixproduct, i) for i = 1:length(tci)]
    return TensorTrain{ValueType,4}(
        [_reshape_splitsites(t, d) for (t, d) in zip(tci, legdims)]
    )
end

"""
    contract_zipup(A::TensorTrain{ValueType,4}, B::TensorTrain{ValueType,4}; tolerance, method, maxbonddim) where {ValueType}

使用 zip-up 算法收缩两个 MPO。

# 参考
https://tensornetwork.org/mps/algorithms/zip_up_mpo/

# 算法
1. 从左到右逐站点收缩
2. 每步使用 SVD 或 LU 分解控制键维度
3. 将另一个因子传递到下一个站点

# 优势
- 边收缩边压缩，避免中间态膨胀
- 精度高于朴素收缩
"""
function contract_zipup(
    A::TensorTrain{ValueType,4},
    B::TensorTrain{ValueType,4};
    tolerance::Float64=1e-12,
    method::Symbol=:SVD,
    maxbonddim::Int=typemax(Int)
) where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end
    
    R::Array{ValueType,3} = ones(ValueType, 1, 1, 1)

    sitetensors = Vector{Array{ValueType,4}}(undef, length(A))
    for n in 1:length(A)
        # 收缩 R 和 A[n]
        RA = _contract(R, A[n], (2,), (1,))

        # 收缩 RA 和 B[n]
        C = permutedims(_contract(RA, B[n], (2, 4), (1, 2)), (1, 2, 4, 3, 5))
        
        if n == length(A)
            sitetensors[n] = reshape(C, size(C)[1:3]..., 1)
            break
        end

        # 分解
        left, right, newbonddim = _factorize(
            reshape(C, prod(size(C)[1:3]), prod(size(C)[4:5])),
            method; tolerance, maxbonddim
        )

        sitetensors[n] = reshape(left, size(C)[1:3]..., newbonddim)
        R = reshape(right, newbonddim, size(C)[4:5]...)
    end

    return TensorTrain{ValueType,4}(sitetensors)
end

# ===================================================================
# 主收缩接口
# ===================================================================

"""
    contract(A::TensorTrain{V1,4}, B::TensorTrain{V2,4}; algorithm=:TCI, tolerance=1e-12, maxbonddim=typemax(Int), f=nothing, kwargs...) where {V1,V2}

收缩两个张量列 A 和 B。

# 参数
- `A, B`: 要收缩的两个张量列（4腿MPO）
- `algorithm`: 收缩算法
  - `:TCI`: 使用 TCI2 拟合收缩结果（推荐）
  - `:naive`: 朴素收缩后 SVD 压缩
  - `:zipup`: 边收缩边分解
- `tolerance`: 容差
- `maxbonddim`: 最大键维度
- `f`: 后处理函数（仅 :TCI 支持）
- `method`: zipup 的分解方法 (:SVD 或 :LU)
- `kwargs...`: 传递给 crossinterpolate2

# 返回值
- 收缩后的张量列

# 示例
```julia
A = TensorTrain{Float64,4}(...)
B = TensorTrain{Float64,4}(...)

# TCI 收缩
C = contract(A, B; algorithm=:TCI, tolerance=1e-10)

# 朴素收缩
C = contract(A, B; algorithm=:naive)

# Zip-up 收缩
C = contract(A, B; algorithm=:zipup)

# 带后处理函数
C = contract(A, B; f=abs2)  # 只有 :TCI 支持
```
"""
function contract(
    A::TensorTrain{V1,4},
    B::TensorTrain{V2,4};
    algorithm::Symbol=:TCI,
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
    f::Union{Nothing,Function}=nothing,
    kwargs...
)::TensorTrain{promote_type(V1,V2),4} where {V1,V2}
    Vres = promote_type(V1, V2)
    A_ = TensorTrain{Vres,4}(A)
    B_ = TensorTrain{Vres,4}(B)
    
    if algorithm === :TCI
        return contract_TCI(A_, B_; tolerance=tolerance, maxbonddim=maxbonddim, f=f, kwargs...)
    elseif algorithm === :naive
        if f !== nothing
            error("Naive contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        return contract_naive(A_, B_; tolerance=tolerance, maxbonddim=maxbonddim)
    elseif algorithm === :zipup
        if f !== nothing
            error("Zipup contraction implementation cannot contract matrix product with a function. Use algorithm=:TCI instead.")
        end
        return contract_zipup(A_, B_; tolerance, maxbonddim)
    else
        throw(ArgumentError("Unknown algorithm $algorithm."))
    end
end

"""
    contract(A::Union{TensorCI1,TensorCI2,TensorTrain{V,3}}, B::TensorTrain{V2,4}; kwargs...)

收缩 MPS 和 MPO。

# 说明
将 3 腿张量列（MPS）转换为 4 腿格式，收缩后再转换回来。
"""
function contract(
    A::Union{TensorCI1{V},TensorCI2{V},TensorTrain{V,3}},
    B::TensorTrain{V2,4};
    kwargs...
)::TensorTrain{promote_type(V,V2),3} where {V,V2}
    tt = contract(TensorTrain{4}(A, [(1, s...) for s in sitedims(A)]), B; kwargs...)
    return TensorTrain{3}(tt, prod.(sitedims(tt)))
end

"""
    contract(A::TensorTrain{V,4}, B::Union{TensorCI1,TensorCI2,TensorTrain{V2,3}}; kwargs...)

收缩 MPO 和 MPS。
"""
function contract(
    A::TensorTrain{V,4},
    B::Union{TensorCI1{V2},TensorCI2{V2},TensorTrain{V2,3}};
    kwargs...
)::TensorTrain{promote_type(V,V2),3} where {V,V2}
    tt = contract(A, TensorTrain{4}(B, [(s..., 1) for s in sitedims(B)]); kwargs...)
    return TensorTrain{3}(tt, prod.(sitedims(tt)))
end
