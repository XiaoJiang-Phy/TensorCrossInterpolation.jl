# ===================================================================
# util.jl - 通用工具函数
# ===================================================================
# 这个文件包含TCI算法中使用的各种辅助函数。
# 这些函数是通用的，可能在多个模块中被调用。
# ===================================================================

"""
    maxabs(maxval::T, updates::AbstractArray{U}) where {T,U}

计算当前最大绝对值与数组中所有元素绝对值的最大值。

# 参数
- `maxval::T`: 当前已知的最大绝对值
- `updates::AbstractArray{U}`: 待比较的数组

# 返回值
- 所有值中绝对值的最大值

# 工作原理
使用reduce函数遍历数组，逐个比较绝对值。
reduce(f, arr; init=val) 从val开始，依次对arr的每个元素应用f。

# 示例
```julia
maxabs(3.0, [-5.0, 2.0, 4.0])  # 返回 5.0
maxabs(10.0, [-5.0, 2.0, 4.0]) # 返回 10.0
```

# 用途
在TCI算法中用于跟踪采样过程中遇到的最大函数值，
以便进行误差归一化。
"""
function maxabs(
    maxval::T,
    updates::AbstractArray{U}
) where {T,U}
    return reduce(
        (x, y) -> max(abs(x), abs(y)),  # 匿名函数：取两个值的绝对值中较大者
        updates[:],                       # 将数组展平为一维（[:]语法）
        init=maxval                       # 初始值
    )
end

"""
    padzero(a::AbstractVector{T}) where {T}

创建一个在原向量后面无限补零的迭代器。

# 参数
- `a::AbstractVector{T}`: 原始向量

# 返回值
- 一个迭代器，先返回a的所有元素，然后无限返回0

# 工作原理
使用Iterators.flatten将两个迭代器连接：
1. 原向量a
2. Iterators.repeated(0) - 无限重复0的迭代器

# 示例
```julia
padded = padzero([1, 2, 3])
collect(Iterators.take(padded, 6))  # 返回 [1, 2, 3, 0, 0, 0]
```

# 用途
在比较不同长度的误差数组时，用零填充较短的数组。
"""
function padzero(a::AbstractVector{T}) where {T}
    return Iterators.flatten((a, Iterators.repeated(0)))
end

"""
    pushunique!(collection, item)

向集合中添加元素，但只在元素不存在时才添加。

# 参数
- `collection`: 任何支持in和push!操作的集合
- `item`: 要添加的元素

# 行为
- 如果item不在collection中，将其添加
- 如果item已存在，不做任何操作

# 示例
```julia
arr = [1, 2, 3]
pushunique!(arr, 4)  # arr变为 [1, 2, 3, 4]
pushunique!(arr, 2)  # arr保持 [1, 2, 3, 4]（2已存在）
```
"""
function pushunique!(collection, item)
    if !(item in collection)  # in运算符检查成员关系
        push!(collection, item)
    end
end

"""
    pushunique!(collection, items...)

向集合中添加多个元素（唯一性保证）。

# 参数
- `collection`: 目标集合
- `items...`: 可变数量的参数，表示要添加的多个元素

# 示例
```julia
arr = [1, 2]
pushunique!(arr, 2, 3, 4)  # arr变为 [1, 2, 3, 4]
```
"""
function pushunique!(collection, items...)
    for item in items
        pushunique!(collection, item)
    end
end

"""
    isconstant(collection)

检查集合中的所有元素是否相同。

# 参数
- `collection`: 任何可迭代的集合

# 返回值
- `true` 如果所有元素相同（或集合为空）
- `false` 如果存在不同的元素

# 示例
```julia
isconstant([1, 1, 1])     # true
isconstant([1, 2, 1])     # false
isconstant([])            # true（空集合被认为是常量）
```
"""
function isconstant(collection)
    if isempty(collection)
        return true
    end
    c = first(collection)  # 获取第一个元素
    return all(collection .== c)  # all检查是否所有条件都为true
end

"""
    randomsubset(set::AbstractArray{ValueType}, n::Int)::Array{ValueType} where {ValueType}

从数组中随机选择n个不重复的元素。

# 参数
- `set::AbstractArray{ValueType}`: 源数组
- `n::Int`: 要选择的元素数量

# 返回值
- 包含n个随机选择元素的新数组

# 行为
- 如果n大于数组长度，返回所有元素（打乱顺序）
- 如果n <= 0，返回空数组
- 选择是无放回的（每个元素最多被选一次）

# 示例
```julia
randomsubset([1, 2, 3, 4, 5], 3)  # 可能返回 [3, 1, 5] 或其他组合
```

# 算法
使用简单的抽取算法：每次随机选一个元素，从源数组中删除，重复n次。
"""
function randomsubset(
    set::AbstractArray{ValueType}, n::Int
)::Array{ValueType} where {ValueType}
    # 限制n不超过集合大小
    n = min(n, length(set))
    if n <= 0
        return ValueType[]  # 返回空数组
    end

    c = copy(set[:])  # 创建副本，避免修改原数组
    subset = Array{ValueType}(undef, n)  # 预分配结果数组
    
    for i in 1:n
        index = rand(1:length(c))  # 随机选择一个索引
        subset[i] = c[index]       # 将选中的元素放入结果
        deleteat!(c, index)        # 从候选集中删除
    end
    return subset
end

"""
    pushrandomsubset!(subset, set, n::Int)

从set中随机选择n个不在subset中的元素，添加到subset中。

# 参数
- `subset`: 目标集合（会被修改）
- `set`: 源集合
- `n::Int`: 要添加的元素数量

# 注意
- 只会添加set中存在但subset中不存在的元素
- setdiff(set, subset) 计算差集

# 示例
```julia
subset = [1, 2]
set = [1, 2, 3, 4, 5]
pushrandomsubset!(subset, set, 2)  # subset可能变为 [1, 2, 4, 3]
```
"""
function pushrandomsubset!(subset, set, n::Int)
    # setdiff计算差集：在set中但不在subset中的元素
    topush = randomsubset(setdiff(set, subset), n)
    append!(subset, topush)  # append!批量添加元素
    nothing  # 返回nothing表示这是一个修改函数
end

"""
    function optfirstpivot(
        f,
        localdims::Union{Vector{Int},NTuple{N,Int}},
        firstpivot::MultiIndex=ones(Int, length(localdims));
        maxsweep=1000
    ) where {N}

优化TCI的初始枢轴点，寻找函数值绝对值较大的点。

# 背景
TCI算法的收敛性很大程度上依赖于初始枢轴点的选择。
理想情况下，初始枢轴应该是函数绝对值最大的点附近。
这个函数使用局部搜索来改进初始猜测。

# 参数
- `f`: 要插值的函数，接受整数索引向量作为输入
- `localdims::Union{Vector{Int},NTuple{N,Int}}`: 各维度的大小
- `firstpivot::MultiIndex`: 起始点，默认为全1
- `maxsweep::Int=1000`: 最大扫描次数

# 返回值
- 优化后的枢轴点（整数索引向量）

# 算法（坐标下降法）
1. 从起始点开始
2. 对每个维度，遍历所有可能的值，选择使|f|最大的值
3. 重复步骤2直到收敛或达到最大迭代次数

# 示例
```julia
f(x) = exp(-sum((x .- 50).^2))  # 高斯函数，峰值在[50,50,...]
localdims = [100, 100, 100]
pivot = optfirstpivot(f, localdims, [1, 1, 1])  # 返回接近[50, 50, 50]的点
```

# 另见
参见: [`crossinterpolate1`](@ref)
"""
function optfirstpivot(
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    firstpivot::Vector{Int}=ones(Int, length(localdims));
    maxsweep=1000
) where {N}
    n = length(localdims)             # 维度数量
    valf = abs(f(firstpivot))         # 当前枢轴的函数绝对值
    pivot = copy(firstpivot)          # 复制以避免修改原始输入

    # TODO: use batch evaluation（待实现：使用批量求值加速）
    
    for _ in 1:maxsweep  # 最多进行maxsweep次扫描
        valf_prev = valf  # 记录本轮开始前的值
        
        # 对每个维度进行优化
        for i in 1:n
            # 尝试该维度的所有可能值
            for d in 1:localdims[i]
                bak = pivot[i]     # 备份当前值
                pivot[i] = d       # 尝试新值
                newval = abs(f(pivot))  # 计算新点的函数绝对值
                
                if newval > valf   # 如果找到更大的值
                    valf = newval  # 更新最大值
                else
                    pivot[i] = bak # 恢复原值
                end
            end
        end
        
        # 如果本轮没有改进，说明已收敛
        if valf_prev == valf
            break
        end
    end

    return pivot
end

"""
    replacenothing(value::Union{T, Nothing}, default::T)::T where {T}

如果值为nothing，返回默认值；否则返回原值。

# 参数
- `value`: 可能为nothing的值
- `default`: 默认值

# 返回值
- 如果value为nothing，返回default；否则返回value

# 示例
```julia
replacenothing(5, 10)       # 返回 5
replacenothing(nothing, 10) # 返回 10
```

# 用途
处理可选参数或查找操作可能返回nothing的情况。
"""
function replacenothing(value::Union{T, Nothing}, default::T)::T where {T}
    if isnothing(value)
        return default
    else
        return value
    end
end


"""
    projector_to_slice(p::AbstractVector{<:Integer})

将投影器向量转换为切片参数。

# 背景
在张量网络中，有时需要对某些索引进行投影（固定到特定值），
而保持其他索引自由。这个函数将投影描述转换为Julia切片语法。

# 参数
- `p::AbstractVector{<:Integer}`: 投影器向量
  - 0 表示该维度保持自由（对应 Colon()）
  - 非0值表示该维度被投影到该值

# 返回值
一个元组 `(slice, shape)`：
- `slice`: 用于数组索引的切片/索引数组
- `shape`: 用于reshape的形状规格

# 示例
```julia
slice, shape = projector_to_slice([0, 3, 0])
# slice = [Colon(), 3, Colon()]  # 对应 A[:, 3, :]
# shape = [Colon(), 1, Colon()]  # 第二维变成1

A = rand(4, 5, 6)
B = A[slice...]           # B的形状是 (4, 1, 6)
C = reshape(B, shape...)  # C的形状仍是 (4, 1, 6)
```

# 说明
构建用于张量核心的切片参数：
- 对于自由索引(0): 使用 Colon() 表示取所有值
- 对于投影索引(非0): 使用具体的索引值
"""
function projector_to_slice(p::AbstractVector{<:Integer})
    # 条件表达式: condition ? value_if_true : value_if_false
    # 如果x==0，则返回Colon()，否则返回x本身
    return [x == 0 ? Colon() : x for x in p], [x == 0 ? Colon() : 1 for x in p]
end