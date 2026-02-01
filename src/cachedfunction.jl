# ===================================================================
# cachedfunction.jl - 带缓存的函数包装器
# ===================================================================
# 这个文件实现了CachedFunction类型，用于缓存函数的计算结果。
#
# 在TCI算法中，同一个索引点可能被多次求值。
# 如果原始函数计算开销大（如物理模拟），缓存可以大幅提高效率。
#
# CachedFunction使用字典存储已计算的值，
# 将多重索引编码为整数作为字典的键，以节省内存和提高查找速度。
# ===================================================================

"""
    struct CachedFunction{ValueType, K} <: BatchEvaluator{ValueType}

带缓存（记忆化）的函数包装器。

# 背景
在TCI算法中，同一个索引点可能被多次访问。
对于计算开销大的函数，缓存已计算的值可以显著提高效率。

# 类型参数
- `ValueType`: 函数返回值的类型
- `K`: 缓存键的整数类型（如UInt32, UInt64, UInt128等）
  - 键类型需要足够大以表示所有可能的索引组合
  - 例如，对于localdims = [10, 10, 10]，需要10^3个键，UInt32足够
  - 对于更大的问题，可能需要UInt128或BitIntegers.UInt256

# 字段
- `f::Function`: 被包装的原始函数
- `localdims::Vector{Int}`: 各维度的大小
- `cache::Dict{K,ValueType}`: 缓存字典
- `coeffs::Vector{K}`: 用于将多重索引编码为整数的系数

# 工作原理
多重索引 [i₁, i₂, ..., iₙ] 被编码为：
key = i₁ + i₂*d₁ + i₃*d₁*d₂ + ...
其中 dₖ 是第k维的大小。这类似于row-major或column-major索引转换。

# 示例
```julia
f(x) = expensive_computation(x)
localdims = [100, 100, 100]
cf = CachedFunction{Float64, UInt64}(f, localdims)

# 第一次调用会计算并缓存
result1 = cf([50, 50, 50])

# 第二次调用直接从缓存返回
result2 = cf([50, 50, 50])  # 立即返回，无需重新计算
```
"""
struct CachedFunction{ValueType,K<:Union{UInt32,UInt64,UInt128,BigInt,BitIntegers.AbstractBitUnsigned}} <: BatchEvaluator{ValueType}
    f::Function                  # 原始函数
    localdims::Vector{Int}       # 各维度的大小 [d₁, d₂, ..., dₙ]
    cache::Dict{K,ValueType}     # 缓存：整数键 -> 函数值
    coeffs::Vector{K}            # 索引编码系数 [1, d₁, d₁*d₂, ...]
    
    """
        CachedFunction{ValueType,K}(f, localdims, cache) where {ValueType,K}
    
    内部构造函数。
    
    计算索引编码系数：coeffs[n] = ∏_{k=1}^{n-1} localdims[k]
    """
    function CachedFunction{ValueType,K}(f, localdims, cache) where {ValueType,K}
        # 计算系数：coeffs[1] = 1, coeffs[n] = localdims[n-1] * coeffs[n-1]
        coeffs = ones(K, length(localdims))
        for n in 2:length(localdims)
            coeffs[n] = localdims[n-1] * coeffs[n-1]
        end
        
        # 警告BigInt的性能问题
        if K == BigInt
            @warn "Using BigInt for keys. This is SUPER slower and uses more memory. The use of BigInt is kept only for compatibility with older code. Use BitIntegers.UInt256 or bigger integer types with fixed size instead."
        end
        
        # 检查溢出：确保最大索引不会超过键类型的范围
        # sum(coeffs .* (localdims .- 1)) 是最大可能的键值
        if K != BigInt
            sum(coeffs .* (localdims .- 1)) < typemax(K) || error("Overflow in CachedFunction. Use ValueType = a bigger type with fixed size, e.g., BitIntegers.UInt256")
        end
        
        new(f, localdims, cache, coeffs)
    end
end

"""
    CachedFunction{ValueType,K}(f, localdims) where {ValueType,K}

创建一个新的CachedFunction（空缓存）。

# 参数
- `ValueType`: 函数返回值类型
- `K`: 键类型（如UInt64）
- `f`: 要包装的函数
- `localdims`: 各维度大小
"""
function CachedFunction{ValueType,K}(f, localdims) where {ValueType,K}
    return CachedFunction{ValueType,K}(f, localdims, Dict{K,ValueType}())
end

"""
    CachedFunction{ValueType}(f, localdims) where {ValueType}

创建CachedFunction，自动选择合适的键类型。

# 自动选择逻辑
根据索引空间大小选择最小的足够大的无符号整数类型：
- UInt32: 最多约40亿个点
- UInt64: 最多约10^19个点
- UInt128: 最多约10^38个点
- UInt256: 更大的空间

# 示例
```julia
cf = CachedFunction{Float64}(f, [10, 10, 10])  # 自动选择UInt32
cf = CachedFunction{Float64}(f, fill(100, 20)) # 可能选择UInt256
```
"""
function CachedFunction{ValueType}(f, localdims) where {ValueType}
    # 计算索引空间的log2大小
    # sum(log2.(big.(localdims))) ≈ log2(∏ dᵢ)
    log2space = sum(log2.(big.(localdims)))
    
    # 根据空间大小选择键类型
    K = if log2space < 31
        UInt32
    elseif log2space < 63
        UInt64
    elseif log2space < 127
        UInt128
    else
        BitIntegers.UInt256
    end
    
    return CachedFunction{ValueType,K}(f, localdims)
end

"""
    localdims(cf::CachedFunction)

获取CachedFunction的局部维度。
"""
function localdims(cf::CachedFunction)
    return cf.localdims
end

"""
    cacheddata(cf::CachedFunction)

获取缓存字典的引用。

# 返回值
- 缓存字典

# 注意
返回的是引用，修改会影响原始缓存。
"""
function cacheddata(cf::CachedFunction)
    return cf.cache
end

"""
    ncacheddata(cf::CachedFunction)

获取已缓存的数据点数量。

# 返回值
- 缓存中的条目数

# 用途
监控缓存使用情况，评估函数调用次数。
"""
function ncacheddata(cf::CachedFunction)
    return length(cf.cache)
end

"""
    key(cf::CachedFunction{V,K}, indexset::MultiIndex) where {V,K}

将多重索引编码为整数键。

# 编码公式
key = ∑ᵢ (indexset[i] - 1) * coeffs[i]

# 参数
- `cf`: CachedFunction对象
- `indexset`: 多重索引 [i₁, i₂, ..., iₙ]

# 返回值
- `K` 类型的整数键

# 注意
索引是1-based，编码时减1使其从0开始。
"""
function key(cf::CachedFunction{V,K}, indexset::MultiIndex) where {V,K}
    # dot product: (indexset .- 1) ⋅ coeffs
    # 使用类型断言确保结果类型正确
    return sum((indexset .- 1) .* cf.coeffs)::K
end

"""
    haskey(cf::CachedFunction, indexset)

检查指定索引是否已在缓存中。

# 返回值
- `true` 如果已缓存，`false` 否则
"""
function Base.haskey(cf::CachedFunction, indexset)
    return key(cf, indexset) ∈ keys(cf.cache)
end

"""
    (cf::CachedFunction{ValueType,K})(indexset::MultiIndex)::ValueType where {ValueType,K}

在指定索引处求值（带缓存）。

# 行为
1. 计算索引的键
2. 如果键在缓存中，直接返回缓存值
3. 否则，调用原始函数，存入缓存，并返回结果

# 参数
- `indexset`: 要求值的索引

# 返回值
- 函数在该索引处的值
"""
function (cf::CachedFunction{ValueType,K})(indexset::MultiIndex)::ValueType where {ValueType,K}
    k = key(cf, indexset)
    
    # get! 是Julia的get-or-compute模式：
    # 如果键存在，返回对应值
    # 如果不存在，调用lambda计算值，存入字典，并返回该值
    return get!(cf.cache, k) do
        cf.f(indexset)
    end
end

"""
    (cf::CachedFunction{ValueType,K})(
        leftindexset::AbstractVector{MultiIndex},
        rightindexset::AbstractVector{MultiIndex},
        ::Val{M}
    )::Array{ValueType,M+2} where {ValueType,K,M}

批量求值接口（带缓存）。

# 实现
对于每个索引组合，检查缓存；如果未命中则计算并缓存。
使用嵌套循环遍历所有组合。
"""
function (cf::CachedFunction{ValueType,K})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M}
)::Array{ValueType,M+2} where {ValueType,K,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{ValueType,M+2}(undef, ntuple(i -> 0, M + 2)...)
    end

    L = length(cf.localdims)
    nl = length(first(leftindexset))
    nr = length(first(rightindexset))

    # 生成中间索引的所有组合
    centerindexset = collect(Iterators.product(ntuple(i -> 1:cf.localdims[nl+i], M)...))
    
    # 预分配结果数组
    result = Array{ValueType,3}(undef, length(leftindexset), length(centerindexset), length(rightindexset))

    for (i, leftindex) in enumerate(leftindexset)
        for (c, centerindices) in enumerate(centerindexset)
            for (j, rightindex) in enumerate(rightindexset)
                # 组装完整索引
                indexset = vcat(leftindex, collect(centerindices), rightindex)
                # 调用带缓存的求值（上面定义的call方法）
                result[i, c, j] = cf(indexset)
            end
        end
    end

    # 重塑结果
    return reshape(result, length(leftindexset), cf.localdims[nl+1:L-nr]..., length(rightindexset))
end

"""
    clearcache!(cf::CachedFunction)

清空缓存。

# 用途
- 释放内存
- 强制重新计算所有值

# 示例
```julia
cf = CachedFunction{Float64}(f, [10, 10, 10])
# ... 使用 cf ...
clearcache!(cf)  # 清空缓存
```
"""
function clearcache!(cf::CachedFunction)
    empty!(cf.cache)
    nothing
end