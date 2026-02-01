# ===================================================================
# globalsearch.jl - 全局误差搜索算法
# ===================================================================
# 这个文件实现了用于估计TCI插值真实误差的全局搜索算法。
#
# 在TCI算法中，局部误差（在枢轴点附近）容易估计，
# 但全局误差（在远离枢轴的区域）可能更大。
# 这些算法通过全局搜索来寻找误差较大的点。
# ===================================================================

"""
    function estimatetrueerror(
        tt::TensorTrain{ValueType,3}, f;
        nsearch::Int = 100,
        initialpoints::Union{Nothing,AbstractVector{MultiIndex}} = nothing,
    )::Vector{Tuple{MultiIndex,Float64}} where {ValueType}

估计张量列近似的真实误差。

# 背景
TCI算法在优化过程中只采样了函数的一小部分点。
插值在这些采样点附近通常很准确，但在远离采样点的区域可能误差较大。
这个函数通过全局搜索来找到误差较大的点。

# 参数
- `tt::TensorTrain{ValueType,3}`: 张量列近似
- `f`: 原始函数（用于计算真实值）
- `nsearch::Int=100`: 搜索的初始点数量
- `initialpoints`: 可选的初始搜索点列表

# 返回值
- `Vector{Tuple{MultiIndex,Float64}}`: 按误差降序排列的(枢轴点, 误差)列表
  - 每个元素是一个元组，包含索引和对应的误差
  - 列表已去重

# 算法
使用"浮动区域"(floating zone)方法：
1. 从多个随机初始点开始
2. 对每个初始点，局部优化以找到误差最大的点
3. 收集并排序所有找到的点

# 示例
```julia
tt = tensortrain(tci)
errors = estimatetrueerror(tt, f; nsearch=50)
if !isempty(errors)
    worstpoint, maxerror = errors[1]
    println("最大误差 \$maxerror 在点 \$worstpoint")
end
```
"""
function estimatetrueerror(
    tt::TensorTrain{ValueType,3}, f;
    nsearch::Int = 100,
    initialpoints::Union{Nothing,AbstractVector{MultiIndex}} = nothing,
)::Vector{Tuple{MultiIndex,Float64}} where {ValueType}
    # 参数验证
    if nsearch <= 0 && initialpoints === nothing
        error("No search is performed")
    end
    nsearch >= 0 || error("nsearch must be non-negative")

    # 如果没有提供初始点，生成随机初始点
    if nsearch > 0 && initialpoints === nothing
        # 对每个维度，随机选择1到该维度最大值之间的整数
        # first(d) 获取sitedims中每个元素的第一个值（对于3维张量，只有一个站点维度）
        initialpoints = [[rand(1:first(d)) for d in sitedims(tt)] for _ in 1:nsearch]
    end

    # 创建带缓存的张量列求值器
    ttcache = TTCache(tt)

    # 对每个初始点，使用浮动区域方法搜索误差最大点
    pivoterror = [_floatingzone(ttcache, f; initp=initp) for initp in initialpoints]

    # 按误差降序排序
    # sortperm返回排序后的索引排列
    # rev=true表示降序
    p = sortperm([e for (_, e) in pivoterror], rev=true)

    # 返回去重后的结果
    return unique(pivoterror[p])
end


"""
    _floatingzone(
        ttcache::TTCache{ValueType}, f;
        earlystoptol::Float64 = typemax(Float64),
        nsweeps=typemax(Int), 
        initp::Union{Nothing,MultiIndex} = nothing
    )::Tuple{MultiIndex,Float64} where {ValueType}

使用浮动区域方法寻找误差最大的点。

# 背景
"浮动区域"是一种局部搜索算法，类似于坐标下降法：
1. 从初始点开始
2. 对每个维度，遍历所有可能的值，选择使误差最大的值
3. 重复直到收敛

# 参数
- `ttcache`: 带缓存的张量列
- `f`: 原始函数
- `earlystoptol`: 提前停止的容差（如果误差超过此值就停止）
- `nsweeps`: 最大扫描次数
- `initp`: 初始点（如果为nothing则随机生成）

# 返回值
- `(pivot, error)` 元组：找到的误差最大点及其误差

# 算法细节
每次扫描遍历所有维度：
1. 固定其他维度，对当前维度计算所有可能值的误差
2. 选择误差最大的值
3. 更新当前点
重复直到误差不再增加（收敛）或达到最大扫描次数。
"""
function _floatingzone(
    ttcache::TTCache{ValueType}, f;
    earlystoptol::Float64 = typemax(Float64),
    nsweeps=typemax(Int), initp::Union{Nothing,MultiIndex} = nothing
)::Tuple{MultiIndex,Float64} where {ValueType}
    nsweeps > 0 || error("nsweeps should be positive!")

    # 获取局部维度
    localdims = first.(sitedims(ttcache))

    n = length(ttcache)

    # 生成或使用初始点
    if initp === nothing
        pivot = [rand(1:d) for d in localdims]
    else
        pivot = initp
    end

    # 计算初始误差
    maxerror = abs(f(pivot) - ttcache(pivot))

    # 迭代优化
    for isweep in 1:nsweeps
        prev_maxerror = maxerror
        
        # 对每个维度进行优化
        for ipos in 1:n
            # 计算当前维度所有可能值的精确函数值
            # filltensor高效地计算所有变体的值
            exactdata = filltensor(
                ValueType,
                f,
                localdims,
                [pivot[1:ipos-1]],   # 左边固定的索引
                [pivot[ipos+1:end]], # 右边固定的索引
                Val(1)               # 只有一个自由维度
            )
            
            # 计算TT近似值
            prediction = filltensor(
                ValueType,
                ttcache,
                localdims,
                [pivot[1:ipos-1]],
                [pivot[ipos+1:end]],
                Val(1)
            )
            
            # 计算逐元素误差
            err = vec(abs.(exactdata .- prediction))
            
            # 更新当前维度的值为误差最大的位置
            pivot[ipos] = argmax(err)
            
            # 更新最大误差
            # 注意：我们取max以确保误差不会因数值问题而减少
            maxerror = max(maximum(err), maxerror)
        end

        # 检查是否收敛或达到提前停止条件
        if maxerror == prev_maxerror || maxerror > earlystoptol
            break
        end
    end

    return pivot, maxerror
end


"""
    fillsitetensors!(tci::TensorCI2{ValueType}, f) where {ValueType}

填充TCI2对象的所有站点张量。

# 参数
- `tci`: TensorCI2对象
- `f`: 用于计算张量元素的函数

# 说明
这个函数用于在需要时重新计算所有站点张量。
在添加全局枢轴后可能需要调用。
"""
function fillsitetensors!(
    tci::TensorCI2{ValueType}, f) where {ValueType}
    for b in 1:length(tci)
       setsitetensor!(tci, f, b)
    end
    nothing
end


"""
    _sanitycheck(tci::TensorCI2{ValueType})::Bool where {ValueType}

对TCI2对象进行完整性检查。

# 检查内容
- 相邻的I集和J集大小是否匹配（确保枢轴矩阵是方阵）

# 返回值
- `true` 如果检查通过
- 抛出错误如果检查失败

# 用途
调试和验证TCI对象的一致性。
"""
function _sanitycheck(tci::TensorCI2{ValueType})::Bool where {ValueType}
    for b in 1:length(tci)-1
        # 第b+1个I集的大小应等于第b个J集的大小（它们定义了第b个枢轴矩阵）
        length(tci.Iset[b+1]) == length(tci.Jset[b]) || error("Pivot matrix at bond $(b) is not square!")
    end

    return true
end