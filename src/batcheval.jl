# ===================================================================
# batcheval.jl - 批量求值接口
# ===================================================================
# 这个文件定义了支持批量函数求值的接口和适配器。
#
# 在TCI算法中，需要频繁地在多个索引点求值。
# 如果函数支持批量求值（一次计算多个点），可以显著提高效率。
# 例如，GPU加速或向量化的函数可以受益于批量求值。
#
# 这个文件提供了：
# 1. BatchEvaluator: 支持批量求值的函数的基类型
# 2. BatchEvaluatorAdapter: 将普通函数包装成批量求值接口
# 3. ThreadedBatchEvaluator: 使用多线程的批量求值器
# ===================================================================

"""
    struct BatchEvaluatorAdapter{T} <: BatchEvaluator{T}

将任何普通函数包装成支持批量求值接口的适配器。

# 背景
TCI算法内部使用批量求值接口。
对于不原生支持批量求值的函数，这个适配器提供兼容性。

# 类型参数
- `T`: 函数的返回值类型

# 字段
- `f::Function`: 被包装的原始函数
- `localdims::Vector{Int}`: 各维度的大小
"""
struct BatchEvaluatorAdapter{T} <: BatchEvaluator{T}
    f::Function
    localdims::Vector{Int}
end

"""
    makebatchevaluatable(::Type{T}, f, localdims) where {T}

将普通函数转换为BatchEvaluatorAdapter。

# 参数
- `T`: 返回值类型
- `f`: 原始函数
- `localdims`: 局部维度

# 返回值
- `BatchEvaluatorAdapter{T}`对象

# 示例
```julia
f(x) = sum(x)
bf = makebatchevaluatable(Float64, f, [10, 10, 10])
bf([1, 2, 3])  # 单点求值
```
"""
makebatchevaluatable(::Type{T}, f, localdims) where {T} = BatchEvaluatorAdapter{T}(f, localdims)

"""
    (bf::BatchEvaluatorAdapter{T})(indexset::MultiIndex)::T where T

单点求值：调用包装的函数。

# 说明
直接调用原始函数，与普通函数调用相同。
"""
function (bf::BatchEvaluatorAdapter{T})(indexset::MultiIndex)::T where T
    bf.f(indexset)
end

"""
    (bf::BatchEvaluatorAdapter{T})(
        leftindexset::AbstractVector{MultiIndex},
        rightindexset::AbstractVector{MultiIndex},
        ::Val{M}
    )::Array{T,M + 2} where {T,M}

批量求值：在所有组合的索引点处求值。

# 参数
- `leftindexset`: 左索引集合（对应张量列的左边部分）
- `rightindexset`: 右索引集合（对应张量列的右边部分）
- `Val{M}`: 中间自由索引的数量（编译时常量）

# 返回值
- `(nleft, d₁, d₂, ..., dₘ, nright)` 形状的数组
  - `nleft = length(leftindexset)`
  - `nright = length(rightindexset)`
  - `d₁, ..., dₘ` 是中间M个维度的大小

# 数学说明
结果数组 R 满足：
R[i, k₁, k₂, ..., kₘ, j] = f([leftindexset[i]..., k₁, k₂, ..., kₘ, rightindexset[j]...])
"""
function (bf::BatchEvaluatorAdapter{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M}
)::Array{T,M + 2} where {T,M}
    # 如果任一索引集为空，返回空数组
    if length(leftindexset) * length(rightindexset) == 0
        # ntuple(d -> 0, M+2) 创建 (0, 0, ..., 0)（M+2个0）
        return Array{T,M + 2}(undef, ntuple(d -> 0, M + 2)...)
    end
    # 委托给通用分发函数
    return _batchevaluate_dispatch(T, bf.f, bf.localdims, leftindexset, rightindexset, Val(M))
end


"""
    _batchevaluate_dispatch(
        ::Type{V},
        f,
        localdims::Vector{Int},
        leftindexset::AbstractVector{MultiIndex},
        rightindexset::AbstractVector{MultiIndex},
        ::Val{M})::Array{V,M + 2} where {V,M}

批量求值的默认实现：使用嵌套循环逐点求值。

# 说明
这是最通用但效率较低的实现。
对于普通函数，简单地遍历所有索引组合并逐个调用函数。

# 实现细节
使用三层嵌套循环：
1. 遍历左索引
2. 遍历中间索引（使用Iterators.product生成笛卡尔积）
3. 遍历右索引
"""
function _batchevaluate_dispatch(
    ::Type{V},
    f,
    localdims::Vector{Int},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M})::Array{V,M + 2} where {V,M}

    # 处理空输入
    if length(leftindexset) * length(rightindexset) == 0
        return Array{V,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    # 计算各部分的长度
    nl = length(first(leftindexset))   # 左索引长度
    nr = length(first(rightindexset))  # 右索引长度
    L = M + nl + nr                     # 总维度数

    # 预分配工作数组
    indexset = MultiIndex(undef, L)
    
    # 预分配结果数组（先用3D，后面reshape）
    # 中间维度被合并，后面再拆分
    result = Array{V,3}(undef, length(leftindexset), prod(localdims[nl+1:L-nr]), length(rightindexset))
    
    # 三层嵌套循环
    for (i, lindex) in enumerate(leftindexset)
        # Iterators.product生成中间索引的所有组合
        # ntuple(x -> 1:localdims[nl+x], M) 创建 (1:d₁, 1:d₂, ..., 1:dₘ)
        for (c, cindex) in enumerate(Iterators.product(ntuple(x -> 1:localdims[nl+x], M)...))
            for (j, rindex) in enumerate(rightindexset)
                # 组装完整索引
                indexset[1:nl] .= lindex
                indexset[nl+1:nl+M] .= cindex
                indexset[nl+M+1:end] .= rindex
                
                # 调用函数并存储结果
                result[i, c, j] = f(indexset)
            end
        end
    end
    
    # 将合并的中间维度拆分回各自的维度
    return reshape(result, length(leftindexset), localdims[nl+1:L-nr]..., length(rightindexset))
end


"""
    _batchevaluate_dispatch(
        ::Type{V},
        f::BatchEvaluator{V},
        localdims::Vector{Int},
        Iset::Vector{MultiIndex},
        Jset::Vector{MultiIndex},
        ::Val{M}
    )::Array{V,M + 2} where {V,M}

当函数本身是BatchEvaluator时的分发实现。

# 说明
如果函数已经支持批量求值（是BatchEvaluator的实例），
直接调用它的批量求值方法，而不是逐点循环。

这利用了Julia的多重派发：相同的函数名，根据参数类型选择不同的实现。
"""
function _batchevaluate_dispatch(
    ::Type{V},
    f::BatchEvaluator{V},
    localdims::Vector{Int},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex},
    ::Val{M}
)::Array{V,M + 2} where {V,M}
    if length(Iset) * length(Jset) == 0
        return Array{V,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    N = length(localdims)
    nl = length(first(Iset))  # 左边索引长度
    nr = length(first(Jset))  # 右边索引长度
    ncent = N - nl - nr       # 中间索引数
    
    # 直接调用BatchEvaluator的批量求值方法
    return f(Iset, Jset, Val(ncent))
end


@doc """
    ThreadedBatchEvaluator{T} <: BatchEvaluator{T}

使用多线程并行化的批量求值器。

# 背景
对于计算密集型的函数，可以使用多线程来加速批量求值。
这个包装器自动将求值任务分配到多个线程。

# 要求
被包装的函数必须是线程安全的（thread-safe）。

# 字段
- `f::Function`: 被包装的函数
- `localdims::Vector{Int}`: 局部维度

# 示例
```julia
# 假设 f 是线程安全的
bf = ThreadedBatchEvaluator{Float64}(f, [10, 10, 10])
# 批量求值将自动使用多线程
result = bf([...], [...], Val(1))
```

# 注意
使用前需要确保Julia启动时有多个线程：
```
julia --threads=8
```
"""
struct ThreadedBatchEvaluator{T} <: BatchEvaluator{T}
    f::Function
    localdims::Vector{Int}
    
    function ThreadedBatchEvaluator{T}(f, localdims) where {T}
        new{T}(f, localdims)
    end
end

"""
    (obj::ThreadedBatchEvaluator{T})(indexset::Vector{Int})::T where {T}

单点求值（直接调用原始函数）。
"""
function (obj::ThreadedBatchEvaluator{T})(indexset::Vector{Int})::T where {T}
    return obj.f(indexset)
end

"""
    (obj::ThreadedBatchEvaluator{T})(
        leftindexset::Vector{Vector{Int}}, 
        rightindexset::Vector{Vector{Int}}, 
        ::Val{M}
    )::Array{T,M + 2} where {T,M}

使用多线程的批量求值。

# 实现
使用 `Threads.@threads` 宏自动并行化循环。
每个线程独立计算一部分索引组合。

# 性能考虑
- 适用于函数计算开销较大的情况
- 对于简单函数，线程开销可能超过收益
"""
function (obj::ThreadedBatchEvaluator{T})(leftindexset::Vector{Vector{Int}}, rightindexset::Vector{Vector{Int}}, ::Val{M})::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M+2}(undef, ntuple(i->0, M+2)...)
    end

    nl = length(first(leftindexset))

    # 生成中间索引的所有组合
    cindexset = vec(collect(Iterators.product(ntuple(i->1:obj.localdims[nl+i], M)...)))
    
    # 生成所有需要计算的索引组合
    elements = collect(Iterators.product(1:length(leftindexset), 1:length(cindexset), 1:length(rightindexset)))
    
    # 预分配结果数组
    result = Array{T,3}(undef, length(leftindexset), length(cindexset), length(rightindexset))

    # 使用多线程并行计算
    # @threads 宏自动将循环分配到可用的线程
    Threads.@threads for indices in elements
        l, c, r = leftindexset[indices[1]], cindexset[indices[2]], rightindexset[indices[3]]
        # vcat连接左、中、右索引
        result[indices...] = obj.f(vcat(l, c..., r))
    end
    
    # 重塑结果数组
    return reshape(result, length(leftindexset), obj.localdims[nl+1:nl+M]..., length(rightindexset))
end