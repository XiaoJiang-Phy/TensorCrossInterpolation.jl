# ===================================================================
# globalpivotfinder.jl - 全局枢轴搜索器
# ===================================================================
# 这个文件实现了用于TCI2算法的全局枢轴搜索功能。
#
# 在TCI2算法中，除了局部的2-site扫描外，还需要偶尔添加"全局枢轴"，
# 即在插值误差较大但不在当前枢轴集覆盖范围内的点。
# 这有助于提高算法的全局收敛性。
# ===================================================================

# 导入随机数相关类型
import Random: AbstractRNG, default_rng

"""
    GlobalPivotSearchInput{ValueType}

全局枢轴搜索算法的输入数据结构。

# 背景
全局枢轴搜索需要访问当前TCI的状态信息。
这个结构体封装了所有必要的信息，使搜索算法与TCI实现解耦。

# 字段
- `localdims::Vector{Int}`: 各维度的大小
- `current_tt::TensorTrain{ValueType,3}`: 当前的张量列近似
- `maxsamplevalue::Float64`: 函数的最大采样值（用于误差归一化）
- `Iset::Vector{Vector{MultiIndex}}`: 当前的左索引集
- `Jset::Vector{Vector{MultiIndex}}`: 当前的右索引集

# 用途
作为全局枢轴搜索算法的输入，包含搜索所需的所有信息。
"""
struct GlobalPivotSearchInput{ValueType}
    localdims::Vector{Int}
    current_tt::TensorTrain{ValueType,3}
    maxsamplevalue::Float64
    Iset::Vector{Vector{MultiIndex}}
    Jset::Vector{Vector{MultiIndex}}

    """
        GlobalPivotSearchInput(
            localdims::Vector{Int},
            current_tt::TensorTrain{ValueType,3},
            maxsamplevalue::ValueType,
            Iset::Vector{Vector{MultiIndex}},
            Jset::Vector{Vector{MultiIndex}}
        ) where {ValueType}

    构造一个GlobalPivotSearchInput对象。
    
    # 参数说明见结构体定义
    """
    function GlobalPivotSearchInput{ValueType}(
        localdims::Vector{Int},
        current_tt::TensorTrain{ValueType,3},
        maxsamplevalue::Float64,
        Iset::Vector{Vector{MultiIndex}},
        Jset::Vector{Vector{MultiIndex}}
    ) where {ValueType}
        new{ValueType}(
            localdims,
            current_tt,
            maxsamplevalue,
            Iset,
            Jset
        )
    end
end


"""
    AbstractGlobalPivotFinder

全局枢轴搜索器的抽象类型。

# 背景
不同的搜索策略可以用不同的具体类型实现。
这个抽象类型定义了所有搜索器的公共接口。

# 接口
所有子类型必须可以作为函数调用（通过定义call方法），
接受GlobalPivotSearchInput和其他参数，返回找到的枢轴列表。
"""
abstract type AbstractGlobalPivotFinder end

"""
    (finder::AbstractGlobalPivotFinder)(
        input::GlobalPivotSearchInput{ValueType},
        f,
        abstol::Float64;
        verbosity::Int=0,
        rng::AbstractRNG=Random.default_rng()
    )::Vector{MultiIndex} where {ValueType}

全局枢轴搜索的接口方法。

# 参数
- `input`: 搜索输入，包含当前TCI状态
- `f`: 原始函数
- `abstol`: 绝对容差，只返回误差大于此值的点
- `verbosity`: 输出详细程度（0=静默，>0=输出调试信息）
- `rng`: 随机数生成器

# 返回值
- `Vector{MultiIndex}`: 找到的需要添加的全局枢轴列表

# 注意
这是一个接口定义。基础实现会抛出错误。
具体的搜索器（如DefaultGlobalPivotFinder）必须实现这个方法。
"""
function (finder::AbstractGlobalPivotFinder)(
    input::GlobalPivotSearchInput{ValueType},
    f,
    abstol::Float64;
    verbosity::Int=0,
    rng::AbstractRNG=Random.default_rng()
)::Vector{MultiIndex} where {ValueType}
    error("find_global_pivots not implemented for $(typeof(finder))")
end

"""
    DefaultGlobalPivotFinder

默认的全局枢轴搜索器，使用随机局部搜索策略。

# 算法
1. 生成nsearch个随机初始点
2. 对每个初始点，进行局部搜索以最大化插值误差
3. 收集误差超过阈值的点
4. 返回最多maxnglobalpivot个枢轴

# 字段
- `nsearch::Int`: 开始搜索的初始点数量
- `maxnglobalpivot::Int`: 每次最多添加的枢轴数量
- `tolmarginglobalsearch::Float64`: 容差边界因子
  - 只返回误差 > abstol * tolmarginglobalsearch 的点
  - 这个边界确保只添加"显著"误差的点

# 示例
```julia
finder = DefaultGlobalPivotFinder(nsearch=10, maxnglobalpivot=5)
pivots = finder(input, f, 1e-8)
```
"""
struct DefaultGlobalPivotFinder <: AbstractGlobalPivotFinder
    nsearch::Int
    maxnglobalpivot::Int
    tolmarginglobalsearch::Float64
end

"""
    DefaultGlobalPivotFinder(;
        nsearch::Int=5,
        maxnglobalpivot::Int=5,
        tolmarginglobalsearch::Float64=10.0
    )

创建一个DefaultGlobalPivotFinder。

# 关键字参数
- `nsearch::Int=5`: 随机初始点的数量
- `maxnglobalpivot::Int=5`: 每次迭代最多添加的枢轴数
- `tolmarginglobalsearch::Float64=10.0`: 容差边界因子
  - 只有误差 > abstol * 10 的点才会被添加
  - 这确保只添加误差显著超过目标容差的点
"""
function DefaultGlobalPivotFinder(;
    nsearch::Int=5,
    maxnglobalpivot::Int=5,
    tolmarginglobalsearch::Float64=10.0
)
    return DefaultGlobalPivotFinder(nsearch, maxnglobalpivot, tolmarginglobalsearch)
end

"""
    (finder::DefaultGlobalPivotFinder)(
        input::GlobalPivotSearchInput{ValueType},
        f,
        abstol::Float64;
        verbosity::Int=0,
        rng::AbstractRNG=Random.default_rng()
    )::Vector{MultiIndex} where {ValueType}

使用随机局部搜索寻找全局枢轴。

# 算法详解
1. 生成nsearch个随机初始点
2. 对每个初始点进行局部搜索：
   a. 对每个维度，遍历所有可能的值
   b. 计算每个变体的插值误差 |f(x) - tt(x)|
   c. 记录找到的最大误差点
3. 过滤：只保留误差 > abstol * tolmarginglobalsearch 的点
4. 截断：最多返回maxnglobalpivot个枢轴

# 参数
- `input`: 包含当前TCI状态的输入
- `f`: 原始函数
- `abstol`: 绝对容差
- `verbosity`: 输出级别（>0时打印找到的枢轴数）
- `rng`: 随机数生成器

# 返回值
- 需要添加的全局枢轴列表
"""
function (finder::DefaultGlobalPivotFinder)(
    input::GlobalPivotSearchInput{ValueType},
    f,
    abstol::Float64;
    verbosity::Int=0,
    rng::AbstractRNG=Random.default_rng()
)::Vector{MultiIndex} where {ValueType}
    L = length(input.localdims)
    nsearch = finder.nsearch
    maxnglobalpivot = finder.maxnglobalpivot
    tolmarginglobalsearch = finder.tolmarginglobalsearch

    # 步骤1: 生成随机初始点
    # rand(rng, 1:d) 在[1,d]范围内生成随机整数
    initial_points = [[rand(rng, 1:input.localdims[p]) for p in 1:L] for _ in 1:nsearch]

    # 步骤2: 对每个初始点进行局部搜索
    found_pivots = MultiIndex[]
    for point in initial_points
        # 局部搜索：尝试改进初始点
        current_point = copy(point)
        best_error = 0.0
        best_point = copy(point)

        # 对每个维度进行优化
        for p in 1:L
            for v in 1:input.localdims[p]
                # 尝试将第p维设为v
                current_point[p] = v
                
                # 计算插值误差
                error = abs(f(current_point) - input.current_tt(current_point))
                
                # 如果找到更大的误差，更新最佳点
                if error > best_error
                    best_error = error
                    best_point = copy(current_point)
                end
            end
            # 恢复原始值（准备优化下一个维度）
            current_point[p] = point[p]
        end

        # 步骤3: 过滤 - 只保留误差足够大的点
        if best_error > abstol * tolmarginglobalsearch
            push!(found_pivots, best_point)
        end
    end

    # 步骤4: 截断 - 限制返回的枢轴数量
    if length(found_pivots) > maxnglobalpivot
        found_pivots = found_pivots[1:maxnglobalpivot]
    end

    # 输出调试信息
    if verbosity > 0
        println("Found $(length(found_pivots)) global pivots")
    end

    return found_pivots
end 