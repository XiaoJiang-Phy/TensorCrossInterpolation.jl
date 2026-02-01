# ===================================================================
# indexset.jl - 索引集数据结构
# ===================================================================
# 这个文件定义了IndexSet类型，用于高效管理多重索引(MultiIndex)的集合。
#
# 在TCI算法中，需要频繁地：
#   - 存储和检索多重索引
#   - 查找某个索引在集合中的位置
#   - 迭代访问所有索引
#
# IndexSet使用双向映射(字典+数组)来实现O(1)的双向查找：
#   - toint: 从索引到整数位置的映射(字典)
#   - fromint: 从整数位置到索引的映射(数组)
# ===================================================================

"""
    struct IndexSet{T}

一个支持双向查找的索引集合数据结构。

# 字段
- `toint::Dict{T,Int}`: 从元素到其位置的映射(哈希表)
- `fromint::Vector{T}`: 从位置到元素的映射(数组)

# 类型参数
- `T`: 存储的元素类型，通常是 `MultiIndex`(即 `Vector{Int}`)

# 设计说明
这种双向映射设计使得：
- 通过位置访问元素: O(1) - 直接数组索引
- 通过元素查找位置: O(1) - 哈希表查找
- 添加新元素: O(1) 平均时间
"""
struct IndexSet{T}
    toint::Dict{T,Int}   # 元素 -> 位置 的映射
    fromint::Vector{T}   # 位置 -> 元素 的映射（有序数组）

    """
        IndexSet{T}() where {T}
    
    创建一个空的IndexSet。
    
    # 示例
    ```julia
    is = IndexSet{Vector{Int}}()  # 创建空的多重索引集
    ```
    """
    function IndexSet{T}() where {T}
        # new{T}是Julia中调用结构体内部构造器的方式
        return new{T}(Dict{T,Int}(), [])
    end

    """
        IndexSet(l::Vector{T}) where {T}
    
    从数组创建IndexSet，数组中的元素顺序决定它们的整数位置。
    
    # 参数
    - `l::Vector{T}`: 初始元素数组
    
    # 示例
    ```julia
    is = IndexSet([[1,2], [3,4], [5,6]])  # 三个多重索引
    # is[1] = [1,2], is[2] = [3,4], is[3] = [5,6]
    ```
    """
    function IndexSet(l::Vector{T}) where {T}
        # 创建字典：将每个元素l[i]映射到其位置i
        # 使用数组推导式(array comprehension)和eachindex遍历索引
        toint = Dict([(l[i], i) for i in eachindex(l)])
        return new{T}(toint, l)
    end
end

"""
    Base.getindex(is::IndexSet{T}, i::Int) where {T}

通过整数索引访问IndexSet中的元素。

# 语法糖
这使得可以使用 `is[i]` 语法访问元素。

# 参数
- `is`: IndexSet对象
- `i::Int`: 整数索引(从1开始)

# 返回值
- 位置i处的元素

# 示例
```julia
is = IndexSet([[1,2], [3,4]])
is[1]  # 返回 [1,2]
is[2]  # 返回 [3,4]
```
"""
function Base.getindex(is::IndexSet{T}, i::Int) where {T}
    return is.fromint[i]
end

"""
    Base.iterate(is::IndexSet{T}) where {T}

实现迭代协议的起始方法。

# 说明
这使得IndexSet可以在for循环中使用。

# 返回值
- 一个元组 `(firstElement, state)`，或者如果为空则返回 `nothing`
"""
function Base.iterate(is::IndexSet{T}) where {T}
    # 委托给内部数组的迭代
    return iterate(is.fromint)
end

"""
    Base.iterate(is::IndexSet{T}, state) where {T}

实现迭代协议的继续方法。

# 参数
- `state`: 上一次迭代返回的状态

# 返回值
- 元组 `(nextElement, newState)`，或者迭代结束时返回 `nothing`
"""
function Base.iterate(is::IndexSet{T}, state) where {T}
    return iterate(is.fromint, state)
end

"""
    pos(is::IndexSet{T}, indices::T) where {T}

查找给定元素在IndexSet中的位置(整数索引)。

# 参数
- `is`: IndexSet对象
- `indices`: 要查找的元素

# 返回值
- 元素的整数位置(1-based)

# 示例
```julia
is = IndexSet([[1,2], [3,4], [5,6]])
pos(is, [3,4])  # 返回 2
```

# 注意
如果元素不存在，将抛出KeyError。
"""
function pos(is::IndexSet{T}, indices::T) where {T}
    return is.toint[indices]
end

"""
    pos(is::IndexSet{T}, indicesvec::AbstractVector{T}) where {T}

批量版本：查找多个元素的位置。

# 参数
- `is`: IndexSet对象
- `indicesvec`: 元素数组

# 返回值
- 整数位置的数组

# 示例
```julia
is = IndexSet([[1,2], [3,4], [5,6]])
pos(is, [[1,2], [5,6]])  # 返回 [1, 3]
```
"""
function pos(is::IndexSet{T}, indicesvec::AbstractVector{T}) where {T}
    # 使用数组推导式对每个元素调用单元素版本
    return [pos(is, i) for i in indicesvec]
end

"""
    Base.setindex!(is::IndexSet{T}, x::T, i::Int) where {T}

设置IndexSet中指定位置的元素。

# 语法糖
这使得可以使用 `is[i] = x` 语法。

# 参数
- `is`: IndexSet对象
- `x`: 新元素
- `i`: 要设置的位置

# 警告
这会覆盖原有的映射关系。如果原位置有元素，
原元素在toint中的映射不会被清除，可能导致不一致。
"""
function Base.setindex!(is::IndexSet{T}, x::T, i::Int) where {T}
    # 更新两个方向的映射
    is.toint[x] = i        # 新元素指向位置i
    is.fromint[i] = x      # 位置i指向新元素
end

"""
    Base.push!(is::IndexSet{T}, x::T) where {T}

向IndexSet末尾添加新元素。

# 语法糖
这使得可以使用 `push!(is, x)` 语法。

# 参数
- `is`: IndexSet对象
- `x`: 要添加的元素

# 示例
```julia
is = IndexSet{Vector{Int}}()
push!(is, [1,2])  # is现在包含 [[1,2]]
push!(is, [3,4])  # is现在包含 [[1,2], [3,4]]
```
"""
function Base.push!(is::IndexSet{T}, x::T) where {T}
    # 先添加到数组末尾
    push!(is.fromint, x)
    # 然后在字典中记录其位置（新的长度）
    is.toint[x] = size(is.fromint, 1)  # size(A, 1)获取第一维的长度
end

"""
    Base.length(is::IndexSet{T}) where {T}

返回IndexSet中的元素数量。

# 返回值
- 整数：元素个数
"""
function Base.length(is::IndexSet{T}) where {T}
    return length(is.fromint)
end

"""
    Base.isempty(is::IndexSet{T}) where {T}

检查IndexSet是否为空。

# 返回值
- `true` 如果为空，`false` 如果非空
"""
function Base.isempty(is::IndexSet{T}) where {T}
    return Base.isempty(is.fromint)
end

"""
    ==(a::IndexSet{T}, b::IndexSet{T}) where {T}

比较两个IndexSet是否相等。

# 说明
两个IndexSet相等当且仅当它们包含相同的元素且顺序相同。

# 注意
只比较fromint数组，因为如果元素相同且顺序相同，
toint字典也必然相同。
"""
function ==(a::IndexSet{T}, b::IndexSet{T}) where {T}
    return a.fromint == b.fromint
end

@doc raw"""
    isnested(a::Vector{T}, b::Vector{T}, row_or_col::Symbol=:row)::Bool where {T}

判断索引集a是否"嵌套"在索引集b中。

# 数学定义
在TCI算法中，"嵌套"(nesting)是一个重要的性质：
- 对于行索引: I_ℓ < I_{ℓ+1} 表示 I_ℓ 中的每个索引都是 I_{ℓ+1} 中某个索引去掉最后一个元素后的结果
- 对于列索引: J_{ℓ+1} < J_ℓ 表示 J_{ℓ+1} 中的每个索引都是 J_ℓ 中某个索引去掉第一个元素后的结果

# 参数
- `a::Vector{T}`: 被检查是否为子集的索引集
- `b::Vector{T}`: 较大的索引集
- `row_or_col::Symbol`: `:row` 检查行嵌套，`:col` 检查列嵌套

# 返回值
- `Bool`: 如果a嵌套在b中返回`true`，否则返回`false`

# 工作原理
- 对于 `:row`: 检查b中每个元素去掉最后一个元素后是否在a中
- 对于 `:col`: 检查b中每个元素去掉第一个元素后是否在a中
"""
function isnested(a::Vector{T}, b::Vector{T}, row_or_col::Symbol=:row)::Bool where {T}
    # 将a转换为集合，便于O(1)的成员检查
    aset = Set(a)
    
    # 遍历b中的每个元素
    for b_ in b
        # 空元素不能嵌套
        if length(b_) == 0
            return false
        end
        
        # 对于行嵌套：b的元素去掉最后一个元素后应该在a中
        # b_[1:end-1] 表示从第1个到倒数第2个元素（Julia中end表示最后一个索引）
        # ∈ 是 in 运算符的数学符号写法
        if row_or_col == :row && !(b_[1:end-1] ∈ aset)
            return false
        end
        
        # 对于列嵌套：b的元素去掉第一个元素后应该在a中
        # b_[2:end] 表示从第2个到最后一个元素
        if row_or_col == :col && !(b_[2:end] ∈ aset)
            return false
        end
    end
    
    return true
end