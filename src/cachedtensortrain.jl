# ===================================================================
# cachedtensortrain.jl - 带缓存的张量列求值器
# ===================================================================
# 这个文件实现了TTCache类型，用于高效地多次求值张量列。
#
# 当需要对同一个张量列进行大量求值时（如在全局搜索中），
# 缓存中间结果可以显著提高效率。
#
# TTCache缓存"左环境"和"右环境"：
#   - 左环境 L[i] = T₁*T₂*...*Tᵢ₋₁ 对给定的索引前缀
#   - 右环境 R[i] = Tᵢ₊₁*...*Tₙ 对给定的索引后缀
# 这样，改变中间某个索引时，可以重用已计算的环境。
# ===================================================================

"""
    abstract type BatchEvaluator{ValueType} end

支持批量求值的函数的抽象基类型。

# 说明
这是所有支持批量求值的类型的抽象父类。
子类型需要实现：
- 单点求值: `(be::BatchEvaluator)(indexset::MultiIndex)`
- 批量求值: `(be::BatchEvaluator)(Iset, Jset, Val(M))`

# 子类型
- [`CachedFunction`](@ref): 带缓存的函数
- [`TTCache`](@ref): 带缓存的张量列
- [`BatchEvaluatorAdapter`](@ref): 普通函数的适配器
"""
abstract type BatchEvaluator{ValueType} end

"""
    struct TTCache{ValueType} <: BatchEvaluator{ValueType}

带缓存的张量列求值器。

# 背景
张量列求值涉及一系列矩阵乘法：
f(i₁,...,iₙ) = T₁[:,i₁,:] * T₂[:,i₂,:] * ... * Tₙ[:,iₙ,:]

当需要大量求值时，可以缓存部分乘积：
- 左环境: L[k] = T₁*T₂*...*Tₖ₋₁ （从左边累积的乘积）
- 右环境: R[k] = Tₖ₊₁*...*Tₙ （从右边累积的乘积）

这样，计算 f(i₁,...,iₙ) = L[k] * Tₖ[:,iₖ,:] * R[k] 可以重用缓存。

# 字段
- `sitetensors::Vector{Array{ValueType,3}}`: 张量核心（重塑为3维）
- `cacheleft::Vector{Dict{MultiIndex,Vector{ValueType}}}`: 左环境缓存
- `cacheright::Vector{Dict{MultiIndex,Vector{ValueType}}}`: 右环境缓存
- `sitedims::Vector{Vector{Int}}`: 各站点的维度

# 缓存结构
- `cacheleft[k][indices]`: 索引 [i₁,...,iₖ₋₁] 对应的左环境向量
- `cacheright[k][indices]`: 索引 [iₖ₊₁,...,iₙ] 对应的右环境向量

# 性能优势
对于n个站点的张量列：
- 无缓存：每次求值 O(n * r²)
- 有缓存：如果只改变一个索引，O(r²)
"""
struct TTCache{ValueType} <: BatchEvaluator{ValueType}
    sitetensors::Vector{Array{ValueType,3}}
    cacheleft::Vector{Dict{MultiIndex,Vector{ValueType}}}
    cacheright::Vector{Dict{MultiIndex,Vector{ValueType}}}
    sitedims::Vector{Vector{Int}}

    """
        TTCache{ValueType}(sitetensors::AbstractVector{<:AbstractArray{ValueType}}, sitedims) where {ValueType}
    
    从张量列表和站点维度创建TTCache。
    """
    function TTCache{ValueType}(sitetensors::AbstractVector{<:AbstractArray{ValueType}}, sitedims) where {ValueType}
        # 检查张量数量和站点维度数量是否匹配
        length(sitetensors) == length(sitedims) || throw(ArgumentError("The number of site tensors and site dimensions must be the same."))
        
        # 检查每个张量的站点维度是否与给定的sitedims一致
        for n in 1:length(sitetensors)
            # prod(sitedims[n]) 是该站点的局部维度乘积
            # size(sitetensors[n])[2:end-1] 的乘积应该相等
            prod(sitedims[n]) == prod(size(sitetensors[n])[2:end-1]) || error("Site dimensions do not match the site tensor dimensions at $n.")
        end

        new{ValueType}(
            # 将每个张量重塑为3维 (左键, 站点, 右键)
            [reshape(x, size(x, 1), :, size(x)[end]) for x in sitetensors],
            # 初始化空的左缓存
            [Dict{MultiIndex,Vector{ValueType}}() for _ in sitetensors],
            # 初始化空的右缓存
            [Dict{MultiIndex,Vector{ValueType}}() for _ in sitetensors],
            sitedims
        )
    end
    
    """
        TTCache{ValueType}(sitetensors::AbstractVector{<:AbstractArray{ValueType}}) where {ValueType}
    
    从张量列表创建TTCache，自动推断站点维度。
    """
    function TTCache{ValueType}(sitetensors::AbstractVector{<:AbstractArray{ValueType}}) where {ValueType}
        return TTCache{ValueType}(sitetensors, [collect(size(x)[2:end-1]) for x in sitetensors])
    end
end

"""
    TTCache(tt::AbstractTensorTrain{V}) where {V}

从张量列创建TTCache。

# 参数
- `tt`: 任何AbstractTensorTrain的实例

# 示例
```julia
tt = tensortrain(tci)
cache = TTCache(tt)
value = cache([1, 2, 3, 4])  # 带缓存的求值
```
"""
function TTCache(tt::AbstractTensorTrain{V}) where {V}
    return TTCache{V}(sitetensors(tt), sitedims(tt))
end

# --- 站点维度相关函数 ---

"""
    sitedims(ttcache::TTCache{ValueType}) where {ValueType}

获取TTCache的站点维度。
"""
function sitedims(ttcache::TTCache{ValueType}) where {ValueType}
    return ttcache.sitedims
end

"""
    length(ttcache::TTCache{ValueType})::Int where {ValueType}

获取TTCache的长度（站点数量）。
"""
function length(ttcache::TTCache{ValueType})::Int where {ValueType}
    return length(ttcache.sitetensors)
end

# --- 左环境计算 ---

"""
    evalleft(ttcache::TTCache{ValueType}, indexset::MultiIndex) where {ValueType}

计算左环境向量（带缓存）。

# 参数
- `ttcache`: TTCache对象
- `indexset`: 左边的索引 [i₁, i₂, ..., iₖ]

# 返回值
- 左环境向量：T₁[:,i₁,:] * T₂[:,i₂,:] * ... * Tₖ[:,iₖ,:]

# 缓存行为
如果索引已在缓存中，直接返回；否则递归计算并缓存。

# 递归结构
evalleft([i₁,...,iₖ]) = evalleft([i₁,...,iₖ₋₁]) * Tₖ[:,iₖ,:]
"""
function evalleft(ttcache::TTCache{ValueType}, indexset::MultiIndex) where {ValueType}
    k = length(indexset)
    
    # 基本情况：空索引
    if k == 0
        return ValueType[1]  # 返回标量1（作为1维向量）
    end

    # 检查缓存
    key = indexset
    if haskey(ttcache.cacheleft[k], key)
        return ttcache.cacheleft[k][key]
    end

    # 递归计算
    # indexset[1:end-1] 是去掉最后一个元素的前缀
    leftenv = evalleft(ttcache, indexset[1:end-1])
    
    # 获取第k个张量在索引 indexset[k] 处的切片
    localtensor = ttcache.sitetensors[k][:, indexset[k], :]
    
    # 矩阵乘法：行向量 * 矩阵 -> 行向量
    # vec 将结果展平为向量
    result = vec(leftenv' * localtensor)

    # 缓存结果
    ttcache.cacheleft[k][key] = result
    return result
end

# --- 右环境计算 ---

"""
    evalright(ttcache::TTCache{ValueType}, indexset::MultiIndex) where {ValueType}

计算右环境向量（带缓存）。

# 参数
- `ttcache`: TTCache对象
- `indexset`: 右边的索引 [iₖ₊₁, ..., iₙ]

# 返回值
- 右环境向量：Tₖ₊₁[:,iₖ₊₁,:] * ... * Tₙ[:,iₙ,:]

# 缓存行为
类似evalleft，使用缓存避免重复计算。

# 递归结构
evalright([iₖ₊₁,...,iₙ]) = Tₖ₊₁[:,iₖ₊₁,:] * evalright([iₖ₊₂,...,iₙ])
"""
function evalright(ttcache::TTCache{ValueType}, indexset::MultiIndex) where {ValueType}
    L = length(ttcache)
    k = L - length(indexset) + 1  # 第一个涉及的张量索引

    # 基本情况：空索引
    if length(indexset) == 0
        return ValueType[1]
    end

    # 检查缓存
    key = indexset
    if haskey(ttcache.cacheright[k], key)
        return ttcache.cacheright[k][key]
    end

    # 递归计算
    # indexset[2:end] 是去掉第一个元素的后缀
    rightenv = evalright(ttcache, indexset[2:end])
    
    # 获取张量切片
    localtensor = ttcache.sitetensors[k][:, indexset[1], :]
    
    # 矩阵乘法：矩阵 * 列向量 -> 列向量
    result = vec(localtensor * rightenv)

    # 缓存结果
    ttcache.cacheright[k][key] = result
    return result
end

# --- 完整求值 ---

"""
    evaluate(ttcache::TTCache{V}, indexset::MultiIndex) where {V}

在指定索引处求值张量列（使用缓存）。

# 计算方法
f(i₁,...,iₙ) = evalleft([i₁,...,iₙ]) （或等价的其他组合）

# 参数
- `indexset`: 完整的索引 [i₁, i₂, ..., iₙ]

# 返回值
- 标量值
"""
function evaluate(ttcache::TTCache{V}, indexset::MultiIndex) where {V}
    # only 从单元素数组中提取标量
    return only(evalleft(ttcache, indexset))
end

"""
    (ttcache::TTCache{V})(indexset::MultiIndex) where {V}

使TTCache可以像函数一样调用。
"""
function (ttcache::TTCache{V})(indexset::MultiIndex) where {V}
    return evaluate(ttcache, indexset)
end

# --- 批量求值 ---

"""
    (ttcache::TTCache{ValueType})(
        leftindexset::AbstractVector{MultiIndex},
        rightindexset::AbstractVector{MultiIndex},
        ::Val{M}
    )::Array{ValueType,M+2} where {ValueType,M}

批量求值接口。

# 说明
对所有左索引、中间索引和右索引的组合进行求值。
利用缓存的左右环境来加速计算。
"""
function (ttcache::TTCache{ValueType})(leftindexset::AbstractVector{MultiIndex}, rightindexset::AbstractVector{MultiIndex}, ::Val{M})::Array{ValueType,M+2} where {ValueType,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{ValueType,M+2}(undef, ntuple(i -> 0, M + 2)...)
    end
    
    L = length(ttcache)
    nl = length(first(leftindexset))  # 左索引长度
    nr = length(first(rightindexset)) # 右索引长度

    # 获取站点维度
    sdims = sitedims(ttcache)
    
    # 计算中间站点的局部维度
    localdims = [prod(d) for d in sdims]
    
    # 生成中间索引的所有组合
    centerindexset = vec(collect(Iterators.product(ntuple(i -> 1:localdims[nl+i], M)...)))

    # 预分配结果
    result = Array{ValueType}(undef, length(leftindexset), length(centerindexset), length(rightindexset))

    for (i, left) in enumerate(leftindexset)
        for (c, center) in enumerate(centerindexset)
            for (j, right) in enumerate(rightindexset)
                # 组装完整索引
                indexset = vcat(left, [center...], right)
                # 利用缓存求值
                result[i, c, j] = ttcache(indexset)
            end
        end
    end

    return reshape(result, length(leftindexset), localdims[nl+1:L-nr]..., length(rightindexset))
end
