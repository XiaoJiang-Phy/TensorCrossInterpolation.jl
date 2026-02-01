# ===================================================================
# tensorci2.jl - TCI2算法实现
# ===================================================================
# 这个文件实现了TensorCI2类型和crossinterpolate2函数。
#
# TCI2（Tensor Cross Interpolation 2）是改进版的张量交叉插值算法。
# 相比TCI1，TCI2具有以下优势：
#   1. 不需要存储完整的Pi矩阵，更节省内存
#   2. 使用2-site更新，收敛更快
#   3. 支持全局枢轴搜索，避免局部极小
#   4. 更好的数值稳定性
#
# TCI2是推荐的默认算法，适用于大多数应用场景。
# ===================================================================

"""
    mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}

TCI2算法产生的张量交叉插值表示。

# 背景
TCI2是改进版的张量交叉插值算法，使用2-site扫描和LU分解
来更高效地选择枢轴。

# 类型参数
- `ValueType`: 张量元素的类型

# 字段
- `Iset::Vector{Vector{MultiIndex}}`: 每个位置的左索引集
- `Jset::Vector{Vector{MultiIndex}}`: 每个位置的右索引集
- `localdims::Vector{Int}`: 各维度的大小
- `sitetensors::Vector{Array{ValueType,3}}`: 站点张量列表
- `pivoterrors::Vector{Float64}`: 回截断的误差估计
- `bonderrors::Vector{Float64}`: 2-site扫描的每键误差
- `maxsamplevalue::Float64`: 误差归一化的最大采样值
- `Iset_history::Vector{Vector{Vector{MultiIndex}}}`: I集的历史（用于非严格嵌套）
- `Jset_history::Vector{Vector{Vector{MultiIndex}}}`: J集的历史

# 创建方式
通常通过 [`crossinterpolate2`](@ref) 函数创建：
```julia
tci, ranks, errors = crossinterpolate2(Float64, f, [10, 10, 10])
```

# 另见
- [`crossinterpolate2`](@ref): TCI2算法的主函数
- [`optimize!`](@ref): TCI2的优化函数
- [`TensorCI1`](@ref): TCI1算法的表示
"""
mutable struct TensorCI2{ValueType} <: AbstractTensorTrain{ValueType}
    Iset::Vector{Vector{MultiIndex}}   # 左索引集
    Jset::Vector{Vector{MultiIndex}}   # 右索引集
    localdims::Vector{Int}             # 局部维度

    sitetensors::Vector{Array{ValueType,3}}  # 站点张量

    "回截断键的误差估计"
    pivoterrors::Vector{Float64}
    "2-site扫描的每键误差"
    bonderrors::Vector{Float64}
    "误差归一化的最大采样值"
    maxsamplevalue::Float64

    # 索引集历史（用于非严格嵌套模式）
    Iset_history::Vector{Vector{Vector{MultiIndex}}}
    Jset_history::Vector{Vector{Vector{MultiIndex}}}

    """
        TensorCI2{ValueType}(localdims) where {ValueType}
    
    创建一个空的TensorCI2对象。
    
    # 参数
    - `localdims`: 各维度的大小（至少2个元素）
    """
    function TensorCI2{ValueType}(
        localdims::Union{Vector{Int},NTuple{N,Int}}
    ) where {ValueType,N}
        length(localdims) > 1 || error("localdims should have at least 2 elements!")
        n = length(localdims)
        new{ValueType}(
            [Vector{MultiIndex}() for _ in 1:n],    # Iset（空）
            [Vector{MultiIndex}() for _ in 1:n],    # Jset（空）
            collect(localdims),                     # localdims
            [zeros(0, d, 0) for d in localdims],    # sitetensors（空）
            [],                                     # pivoterrors
            zeros(length(localdims) - 1),           # bonderrors
            0.0,                                    # maxsamplevalue
            Vector{Vector{MultiIndex}}[],           # Iset_history
            Vector{Vector{MultiIndex}}[],           # Jset_history
        )
    end
end

"""
    TensorCI2{ValueType}(func, localdims, initialpivots) where {ValueType}

从函数和初始枢轴创建TensorCI2。

# 参数
- `func`: 要插值的函数
- `localdims`: 各维度大小
- `initialpivots`: 初始枢轴列表
"""
function TensorCI2{ValueType}(
    func::F,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))]
) where {F,ValueType,N}
    tci = TensorCI2{ValueType}(localdims)
    addglobalpivots!(tci, initialpivots)
    tci.maxsamplevalue = maximum(abs, (func(x) for x in initialpivots))
    abs(tci.maxsamplevalue) > 0.0 || error("maxsamplevalue is zero!")
    invalidatesitetensors!(tci)
    return tci
end

"""
    TensorCI2{ValueType}(func, localdims, Iset, Jset) where {ValueType}

从局部枢轴列表初始化TCI2对象。
"""
function TensorCI2{ValueType}(
    func::F,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    Iset::Vector{Vector{MultiIndex}},
    Jset::Vector{Vector{MultiIndex}}
) where {F,ValueType,N}
    tci = TensorCI2{ValueType}(localdims)
    tci.Iset = Iset
    tci.Jset = Jset
    pivots = reconstractglobalpivotsfromijset(localdims, tci.Iset, tci.Jset)
    tci.maxsamplevalue = maximum(abs, (func(bit) for bit in pivots))
    abs(tci.maxsamplevalue) > 0.0 || error("maxsamplevalue is zero!")
    invalidatesitetensors!(tci)
    return tci
end

# ===================================================================
# 嵌套性检查
# ===================================================================

@doc raw"""
    printnestinginfo(tci::TensorCI2{T}) where {T}

打印嵌套条件的满足情况。

# 背景
TCI理论要求索引集满足嵌套条件：
- I_ℓ ⊂ I_{ℓ+1}（左索引递增嵌套）
- J_{ℓ+1} ⊂ J_ℓ（右索引递减嵌套）

这个函数检查并输出每对键的嵌套情况。
"""
function printnestinginfo(tci::TensorCI2{T}) where {T}
    printnestinginfo(stdout, tci)
end

function printnestinginfo(io::IO, tci::TensorCI2{T}) where {T}
    println(io, "Nesting info: Iset")
    for i in 1:length(tci.Iset)-1
        if isnested(tci.Iset[i], tci.Iset[i+1], :row)
            println(io, "  Nested: $(i) < $(i+1)")
        else
            println(io, "  Not nested: $(i) !< $(i+1)")
        end
    end

    println(io)
    println(io, "Nesting info: Jset")
    for i in 1:length(tci.Jset)-1
        if isnested(tci.Jset[i+1], tci.Jset[i], :col)
            println(io, "  Nested: $(i+1) < $i")
        else
            println(io, "  Not nested: ! $(i+1) < $i")
        end
    end
end

# ===================================================================
# 维度相关函数
# ===================================================================

"""
    linkdims(tci::TensorCI2{T}) where {T}

获取所有键维度。
"""
function linkdims(tci::TensorCI2{T})::Vector{Int} where {T}
    return [length(tci.Iset[b+1]) for b in 1:length(tci)-1]
end

# ===================================================================
# 站点张量管理
# ===================================================================

"""
    invalidatesitetensors!(tci::TensorCI2{T}) where {T}

使所有站点张量无效。

# 说明
当索引集改变后，站点张量需要重新计算。
这个函数将张量设为空，标记它们需要更新。
"""
function invalidatesitetensors!(tci::TensorCI2{T}) where {T}
    for b in 1:length(tci)
        tci.sitetensors[b] = zeros(T, 0, 0, 0)
    end
    nothing
end

"""
    issitetensorsavailable(tci::TensorCI2{T}) where {T}

检查站点张量是否可用。
"""
function issitetensorsavailable(tci::TensorCI2{T}) where {T}
    return all(length(tci.sitetensors[b]) != 0 for b in 1:length(tci))
end

# ===================================================================
# 误差管理
# ===================================================================

"""
    updatebonderror!(tci::TensorCI2{T}, b::Int, error::Float64) where {T}

更新键 b 的误差。
"""
function updatebonderror!(
    tci::TensorCI2{T}, b::Int, error::Float64
) where {T}
    tci.bonderrors[b] = error
    nothing
end

"""
    maxbonderror(tci::TensorCI2{T}) where {T}

获取最大键误差。
"""
function maxbonderror(tci::TensorCI2{T}) where {T}
    return maximum(tci.bonderrors)
end

"""
    updatepivoterror!(tci::TensorCI2{T}, errors::AbstractVector{Float64}) where {T}

更新枢轴误差数组。
"""
function updatepivoterror!(tci::TensorCI2{T}, errors::AbstractVector{Float64}) where {T}
    # 用当前误差和新误差的最大值更新
    erroriter = Iterators.map(max, padzero(tci.pivoterrors), padzero(errors))
    tci.pivoterrors = Iterators.take(
        erroriter,
        max(length(tci.pivoterrors), length(errors))
    ) |> collect
    nothing
end

"""
    flushpivoterror!(tci::TensorCI2{T}) where {T}

清空枢轴误差数组。
"""
function flushpivoterror!(tci::TensorCI2{T}) where {T}
    tci.pivoterrors = Float64[]
    nothing
end

"""
    pivoterror(tci::TensorCI2{T}) where {T}

获取当前枢轴误差。
"""
function pivoterror(tci::TensorCI2{T}) where {T}
    return maxbonderror(tci)
end

function updateerrors!(
    tci::TensorCI2{T},
    b::Int,
    errors::AbstractVector{Float64}
) where {T}
    updatebonderror!(tci, b, last(errors))
    updatepivoterror!(tci, errors)
    nothing
end

# ===================================================================
# 全局枢轴管理
# ===================================================================

"""
    reconstractglobalpivotsfromijset(localdims, Isets, Jsets)

从I集和J集重构全局枢轴。

# 说明
给定I集和J集，计算对应的完整多索引列表。
"""
function reconstractglobalpivotsfromijset(
    localdims::Union{Vector{Int},NTuple{N,Int}},
    Isets::Vector{Vector{MultiIndex}}, 
    Jsets::Vector{Vector{MultiIndex}}
) where {N}
    pivots = []
    l = length(Isets)
    for i in 1:l
        for Iset in Isets[i]
            for Jset in Jsets[i]
                for j in 1:localdims[i]
                    pushunique!(pivots, vcat(Iset, [j], Jset))
                end
            end
        end
    end
    return pivots
end

"""
    addglobalpivots!(tci::TensorCI2{ValueType}, pivots::Vector{MultiIndex}) where {ValueType}

添加全局枢轴到索引集。

# 参数
- `tci`: TensorCI2对象
- `pivots`: 要添加的全局枢轴列表

# 说明
每个全局枢轴被分解为左索引和右索引，分别添加到Iset和Jset中。
添加后站点张量会被标记为无效。
"""
function addglobalpivots!(
    tci::TensorCI2{ValueType},
    pivots::Vector{MultiIndex}
) where {ValueType}
    # 验证枢轴长度
    if any(length(tci) .!= length.(pivots))
        throw(DimensionMismatch("Please specify a pivot as one index per leg of the MPS."))
    end

    for pivot in pivots
        for b in 1:length(tci)
            # 将枢轴分解为左右索引
            pushunique!(tci.Iset[b], pivot[1:b-1])
            pushunique!(tci.Jset[b], pivot[b+1:end])
        end
    end

    if length(pivots) > 0
        invalidatesitetensors!(tci)
    end

    nothing
end

"""
    addglobalpivots1sitesweep!(tci, f, pivots; kwargs...)

添加全局枢轴并执行1-site扫描。

# 说明
添加枢轴后，执行makecanonical!来更新索引集。
"""
function addglobalpivots1sitesweep!(
    tci::TensorCI2{ValueType},
    f::F,
    pivots::Vector{MultiIndex};
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim=typemax(Int)
) where {F,ValueType}
    addglobalpivots!(tci, pivots)
    makecanonical!(tci, f; reltol=reltol, abstol=abstol, maxbonddim=maxbonddim)
end

"""
    existaspivot(tci::TensorCI2{ValueType}, indexset::MultiIndex) where {ValueType}

检查索引是否已存在于枢轴集中。
"""
function existaspivot(
    tci::TensorCI2{ValueType},
    indexset::MultiIndex) where {ValueType}
    return [indexset[1:b-1] ∈ tci.Iset[b] && indexset[b+1:end] ∈ tci.Jset[b] for b in 1:length(tci)]
end

"""
    addglobalpivots2sitesweep!(tci, f, pivots; kwargs...)

添加全局枢轴并执行2-site扫描。

# 参数
- `tci`: TensorCI2对象
- `f`: 原始函数
- `pivots`: 要添加的枢轴列表
- `ntry`: 最大尝试次数

# 返回值
- 未成功添加的枢轴数量

# 说明
重复尝试添加枢轴直到所有枢轴都被添加或达到最大尝试次数。
"""
function addglobalpivots2sitesweep!(
    tci::TensorCI2{ValueType},
    f::F,
    pivots::Vector{MultiIndex};
    tolerance::Float64=1e-8,
    normalizeerror::Bool=true,
    maxbonddim=typemax(Int),
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    ntry::Int=10,
    strictlynested::Bool=false
)::Int where {F,ValueType}
    if any(length(tci) .!= length.(pivots))
        throw(DimensionMismatch("Please specify a pivot as one index per leg of the MPS."))
    end

    pivots_ = pivots

    for _ in 1:ntry
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        abstol = tolerance * errornormalization

        addglobalpivots!(tci, pivots_)

        sweep2site!(
            tci, f, 2;
            abstol=abstol,
            maxbonddim=maxbonddim,
            pivotsearch=pivotsearch,
            strictlynested=strictlynested,
            verbosity=verbosity)

        # 检查哪些枢轴还没有被正确添加
        newpivots = [p for p in pivots if abs(evaluate(tci, p) - f(p)) > abstol]

        if verbosity > 0
            println("Trying to add $(length(pivots_)) global pivots, $(length(newpivots)) still remain.")
        end

        if length(newpivots) == 0 || Set(newpivots) == Set(pivots_)
            return length(newpivots)
        end

        pivots_ = newpivots
    end
    return length(pivots_)
end

# ===================================================================
# 张量填充函数
# ===================================================================

"""
    filltensor(::Type{ValueType}, f, localdims, Iset, Jset, ::Val{M}) where {ValueType,M}

填充M+2维张量。

# 参数
- `ValueType`: 元素类型
- `f`: 函数
- `localdims`: 局部维度
- `Iset`: 左索引集
- `Jset`: 右索引集
- `Val{M}`: 中间自由索引的数量

# 返回值
形状为 (|Iset|, d₁, ..., dₘ, |Jset|) 的数组
"""
function filltensor(
    ::Type{ValueType},
    f,
    localdims::Vector{Int},
    Iset::Vector{MultiIndex},
    Jset::Vector{MultiIndex},
    ::Val{M}
)::Array{ValueType,M+2} where {ValueType,M}
    if length(Iset) * length(Jset) == 0
        return Array{ValueType,M+2}(undef, ntuple(i->0, M+2)...)
    end

    N = length(localdims)
    nl = length(first(Iset))
    nr = length(first(Jset))
    ncent = N - nl - nr
    expected_size = (length(Iset), localdims[nl+1:nl+ncent]..., length(Jset))
    M == ncent || error("Invalid number of central indices")
    return reshape(
        _batchevaluate_dispatch(ValueType, f, localdims, Iset, Jset, Val(ncent)),
        expected_size...
    )
end

# ===================================================================
# 索引集笛卡尔积
# ===================================================================

"""
    kronecker(Iset, localdim)

计算索引集和局部索引的笛卡尔积（追加在右边）。

# 示例
如果 Iset = [[1], [2]]，localdim = 3，
则返回 [[1,1], [1,2], [1,3], [2,1], [2,2], [2,3]]
"""
function kronecker(
    Iset::Union{Vector{MultiIndex},IndexSet{MultiIndex}},
    localdim::Int
)
    return MultiIndex[[is..., j] for is in Iset, j in 1:localdim][:]
end

"""
    kronecker(localdim, Jset)

计算局部索引和索引集的笛卡尔积（追加在左边）。
"""
function kronecker(
    localdim::Int,
    Jset::Union{Vector{MultiIndex},IndexSet{MultiIndex}}
)
    return MultiIndex[[i, js...] for i in 1:localdim, js in Jset][:]
end

"""
    setsitetensor!(tci::TensorCI2{ValueType}, b::Int, T::AbstractArray{ValueType,N}) where {ValueType,N}

设置位置 b 的站点张量。
"""
function setsitetensor!(
    tci::TensorCI2{ValueType}, b::Int, T::AbstractArray{ValueType,N}
) where {ValueType,N}
    tci.sitetensors[b] = reshape(
        T,
        length(tci.Iset[b]),
        tci.localdims[b],
        length(tci.Jset[b])
    )
end

# ===================================================================
# 0-site扫描（坏枢轴移除）
# ===================================================================

"""
    sweep0site!(tci::TensorCI2{ValueType}, f, b::Int; reltol, abstol) where {ValueType}

在键 b 执行0-site扫描（移除坏枢轴）。

# 说明
通过对P矩阵进行LU分解来识别并移除数值上较差的枢轴。
"""
function sweep0site!(
    tci::TensorCI2{ValueType}, f, b::Int;
    reltol=1e-14, abstol=0.0
) where {ValueType}
    invalidatesitetensors!(tci)
    
    # 构建枢轴矩阵
    P = reshape(
        filltensor(ValueType, f, tci.localdims, tci.Iset[b+1], tci.Jset[b], Val(0)),
        length(tci.Iset[b+1]), length(tci.Jset[b]))
    updatemaxsample!(tci, P)
    
    # LU分解
    F = MatrixLUCI(
        P,
        reltol=reltol,
        abstol=abstol,
        leftorthogonal=true
    )

    # 计算有多少对角元素满足阈值
    ndiag = sum(abs(F.lu.U[i,i]) > abstol && abs(F.lu.U[i,i]/F.lu.U[1,1]) > reltol for i in eachindex(diag(F.lu.U)))

    # 截断索引集
    tci.Iset[b+1] = tci.Iset[b+1][rowindices(F)[1:ndiag]]
    tci.Jset[b] = tci.Jset[b][colindices(F)[1:ndiag]]
    return nothing
end

# 向后兼容
const rmbadpivots! = sweep0site!

"""
    setsitetensor!(tci::TensorCI2{ValueType}, f, b::Int; leftorthogonal=true) where {ValueType}

计算并设置位置 b 的站点张量。

# 说明
使用函数求值填充Pi矩阵，然后通过解线性方程组计算站点张量。
"""
function setsitetensor!(
    tci::TensorCI2{ValueType}, f, b::Int; leftorthogonal=true
) where {ValueType}
    leftorthogonal || error("leftorthogonal==false is not supported!")

    # 计算包含站点维度的Pi矩阵
    Is = leftorthogonal ? kronecker(tci.Iset[b], tci.localdims[b]) : tci.Iset[b]
    Js = leftorthogonal ? tci.Jset[b] : kronecker(tci.localdims[b], tci.Jset[b])
    Pi1 = reshape(
        filltensor(ValueType, f, tci.localdims, tci.Iset[b], tci.Jset[b], Val(1)),
        length(Is), length(Js))
    updatemaxsample!(tci, Pi1)

    # 边界情况特殊处理
    if (leftorthogonal && b == length(tci)) ||
        (!leftorthogonal && b == 1)
        setsitetensor!(tci, b, Pi1)
        return tci.sitetensors[b]
    end

    # 计算枢轴矩阵P
    P = reshape(
        filltensor(ValueType, f, tci.localdims, tci.Iset[b+1], tci.Jset[b], Val(0)),
        length(tci.Iset[b+1]), length(tci.Jset[b]))
    length(tci.Iset[b+1]) == length(tci.Jset[b]) || error("Pivot matrix at bond $(b) is not square!")

    # 计算 T = Pi1 * P^{-1}
    Tmat = transpose(transpose(P) \ transpose(Pi1))
    tci.sitetensors[b] = reshape(Tmat, length(tci.Iset[b]), tci.localdims[b], length(tci.Iset[b+1]))
    return tci.sitetensors[b]
end

"""
    updatemaxsample!(tci::TensorCI2{V}, samples::Array{V}) where {V}

更新最大采样值。
"""
function updatemaxsample!(tci::TensorCI2{V}, samples::Array{V}) where {V}
    tci.maxsamplevalue = maxabs(tci.maxsamplevalue, samples)
end

# ===================================================================
# 1-site扫描
# ===================================================================

"""
    sweep1site!(tci::TensorCI2{ValueType}, f, sweepdirection; kwargs...) where {ValueType}

执行1-site扫描。

# 参数
- `sweepdirection`: 扫描方向 (:forward 或 :backward)
- `reltol, abstol`: 容差
- `maxbonddim`: 最大键维度
- `updatetensors`: 是否更新张量

# 说明
1-site扫描对每个站点单独进行LU分解，更新索引集。
这比2-site扫描更快但精度可能略低。
"""
function sweep1site!(
    tci::TensorCI2{ValueType},
    f,
    sweepdirection::Symbol=:forward;
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim::Int=typemax(Int),
    updatetensors::Bool=true
) where {ValueType}
    flushpivoterror!(tci)
    invalidatesitetensors!(tci)

    if !(sweepdirection === :forward || sweepdirection === :backward)
        throw(ArgumentError("Unknown sweep direction $sweepdirection: choose between :forward, :backward."))
    end

    forwardsweep = sweepdirection === :forward
    
    for b in (forwardsweep ? (1:length(tci)-1) : (length(tci):-1:2))
        # 根据方向选择索引集
        Is = forwardsweep ? kronecker(tci.Iset[b], tci.localdims[b]) : tci.Iset[b]
        Js = forwardsweep ? tci.Jset[b] : kronecker(tci.localdims[b], tci.Jset[b])
        
        # 构建并分解Pi矩阵
        Pi = reshape(
            filltensor(ValueType, f, tci.localdims, tci.Iset[b], tci.Jset[b], Val(1)),
            length(Is), length(Js))
        updatemaxsample!(tci, Pi)
        
        luci = MatrixLUCI(
            Pi,
            reltol=reltol,
            abstol=abstol,
            maxrank=maxbonddim,
            leftorthogonal=forwardsweep
        )
        
        # 更新索引集
        tci.Iset[b+forwardsweep] = Is[rowindices(luci)]
        tci.Jset[b-!forwardsweep] = Js[colindices(luci)]
        
        # 更新张量
        if updatetensors
            setsitetensor!(tci, b, forwardsweep ? left(luci) : right(luci))
        end
        
        if any(isnan.(tci.sitetensors[b]))
            error("Error: NaN in tensor T[$b]")
        end
        
        updateerrors!(tci, b - !forwardsweep, pivoterrors(luci))
    end

    # 更新最后一个张量
    if updatetensors
        lastupdateindex = forwardsweep ? length(tci) : 1
        shape = if forwardsweep
            (length(tci.Iset[end]), tci.localdims[end])
        else
            (tci.localdims[begin], length(tci.Jset[begin]))
        end
        localtensor = reshape(filltensor(
            ValueType, f, tci.localdims, tci.Iset[lastupdateindex], tci.Jset[lastupdateindex], Val(1)), shape)
        setsitetensor!(tci, lastupdateindex, localtensor)
    end
    nothing
end

"""
    makecanonical!(tci::TensorCI2{ValueType}, f; kwargs...) where {ValueType}

将TCI2转换为规范形式。

# 说明
执行三次1-site扫描：
1. 正向（无压缩）
2. 反向（压缩）
3. 正向（压缩并计算张量）
"""
function makecanonical!(
    tci::TensorCI2{ValueType},
    f::F;
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim::Int=typemax(Int)
) where {F,ValueType}
    # 第一个半扫描精确进行，不压缩
    sweep1site!(tci, f, :forward; reltol=0.0, abstol=0.0, maxbonddim=typemax(Int), updatetensors=false)
    sweep1site!(tci, f, :backward; reltol, abstol, maxbonddim, updatetensors=false)
    sweep1site!(tci, f, :forward; reltol, abstol, maxbonddim, updatetensors=true)
end

# ===================================================================
# 2-site扫描辅助类型
# ===================================================================

"""
    SubMatrix{T}

用于惰性求值的子矩阵类型。

# 说明
在车象搜索中，我们不需要计算整个Pi矩阵，
只需要按需计算被访问的元素。
"""
mutable struct SubMatrix{T}
    f::Function               # 原始函数
    rows::Vector{MultiIndex}  # 行索引
    cols::Vector{MultiIndex}  # 列索引
    maxsamplevalue::Float64   # 最大采样值

    function SubMatrix{T}(f, rows, cols) where {T}
        new(f, rows, cols, 0.0)
    end
end

"""
    _submatrix_batcheval(obj::SubMatrix{T}, f, irows, icols) where {T}

子矩阵的批量求值（普通函数版本）。
"""
function _submatrix_batcheval(obj::SubMatrix{T}, f, irows::Vector{Int}, icols::Vector{Int})::Matrix{T} where {T}
    return [f(vcat(obj.rows[i], obj.cols[j])) for i in irows, j in icols]
end

"""
    _submatrix_batcheval(obj::SubMatrix{T}, f::BatchEvaluator{T}, irows, icols) where {T}

子矩阵的批量求值（BatchEvaluator版本）。
"""
function _submatrix_batcheval(obj::SubMatrix{T}, f::BatchEvaluator{T}, irows::Vector{Int}, icols::Vector{Int})::Matrix{T} where {T}
    Iset = [obj.rows[i] for i in irows]
    Jset = [obj.cols[j] for j in icols]
    return f(Iset, Jset, Val(0))
end

"""
    (obj::SubMatrix{T})(irows::Vector{Int}, icols::Vector{Int}) where {T}

使SubMatrix可调用。
"""
function (obj::SubMatrix{T})(irows::Vector{Int}, icols::Vector{Int})::Matrix{T} where {T}
    res = _submatrix_batcheval(obj, obj.f, irows, icols)
    obj.maxsamplevalue = max(obj.maxsamplevalue, maximum(abs, res))
    return res
end

# ===================================================================
# 2-site枢轴更新
# ===================================================================

"""
    updatepivots!(tci::TensorCI2{ValueType}, b::Int, f, leftorthogonal; kwargs...) where {ValueType}

使用TCI2算法更新键 b 的枢轴。

# 参数
- `b`: 键索引
- `f`: 原始函数
- `leftorthogonal`: 是否左正交化
- `pivotsearch`: 枢轴搜索策略 (:full 或 :rook)
- `extraIset, extraJset`: 额外的索引（用于非严格嵌套）

# 说明
这是TCI2的核心更新步骤。
"""
function updatepivots!(
    tci::TensorCI2{ValueType},
    b::Int,
    f::F,
    leftorthogonal::Bool;
    reltol::Float64=1e-14,
    abstol::Float64=0.0,
    maxbonddim::Int=typemax(Int),
    sweepdirection::Symbol=:forward,
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    extraIset::Vector{MultiIndex}=MultiIndex[],
    extraJset::Vector{MultiIndex}=MultiIndex[],
) where {F,ValueType}
    invalidatesitetensors!(tci)

    # 组合当前索引和额外索引
    Icombined = union(kronecker(tci.Iset[b], tci.localdims[b]), extraIset)
    Jcombined = union(kronecker(tci.localdims[b+1], tci.Jset[b+1]), extraJset)

    luci = if pivotsearch === :full
        # 完全搜索：计算整个Pi矩阵
        t1 = time_ns()
        Pi = reshape(
            filltensor(ValueType, f, tci.localdims,
            Icombined, Jcombined, Val(0)),
            length(Icombined), length(Jcombined)
        )
        t2 = time_ns()

        updatemaxsample!(tci, Pi)
        luci = MatrixLUCI(
            Pi,
            reltol=reltol,
            abstol=abstol,
            maxrank=maxbonddim,
            leftorthogonal=leftorthogonal
        )
        t3 = time_ns()
        if verbosity > 2
            x, y = length(Icombined), length(Jcombined)
            println("    Computing Pi ($x x $y) at bond $b: $(1e-9*(t2-t1)) sec, LU: $(1e-9*(t3-t2)) sec")
        end
        luci
    elseif pivotsearch === :rook
        # 车象搜索：惰性求值Pi矩阵元素
        t1 = time_ns()
        I0 = Int.(Iterators.filter(!isnothing, findfirst(isequal(i), Icombined) for i in tci.Iset[b+1]))::Vector{Int}
        J0 = Int.(Iterators.filter(!isnothing, findfirst(isequal(j), Jcombined) for j in tci.Jset[b]))::Vector{Int}
        Pif = SubMatrix{ValueType}(f, Icombined, Jcombined)
        t2 = time_ns()
        res = MatrixLUCI(
            ValueType,
            Pif,
            (length(Icombined), length(Jcombined)),
            I0, J0;
            reltol=reltol, abstol=abstol,
            maxrank=maxbonddim,
            leftorthogonal=leftorthogonal,
            pivotsearch=:rook,
            usebatcheval=true
        )
        updatemaxsample!(tci, [ValueType(Pif.maxsamplevalue)])

        t3 = time_ns()

        # 如果车象搜索失败，回退到完全搜索
        if npivots(res) == 0
            Pi = reshape(
                filltensor(ValueType, f, tci.localdims,
                Icombined, Jcombined, Val(0)),
                length(Icombined), length(Jcombined)
            )
            updatemaxsample!(tci, Pi)
            res = MatrixLUCI(
                Pi,
                reltol=reltol,
                abstol=abstol,
                maxrank=maxbonddim,
                leftorthogonal=leftorthogonal
            )
        end

        t4 = time_ns()
        if verbosity > 2
            x, y = length(Icombined), length(Jcombined)
            println("    Computing Pi ($x x $y) at bond $b: $(1e-9*(t2-t1)) sec, LU: $(1e-9*(t3-t2)) sec, fall back to full: $(1e-9*(t4-t3)) sec")
        end
        res
    else
        throw(ArgumentError("Unknown pivot search strategy $pivotsearch. Choose from :rook, :full."))
    end
    
    # 更新索引集
    tci.Iset[b+1] = Icombined[rowindices(luci)]
    tci.Jset[b] = Jcombined[colindices(luci)]
    
    # 更新站点张量
    if length(extraIset) == 0 && length(extraJset) == 0
        setsitetensor!(tci, b, left(luci))
        setsitetensor!(tci, b + 1, right(luci))
    end
    
    updateerrors!(tci, b, pivoterrors(luci))
    nothing
end

# ===================================================================
# 收敛判断
# ===================================================================

"""
    convergencecriterion(ranks, errors, nglobalpivots, tolerance, maxbonddim, ncheckhistory; checkconvglobalpivot) -> Bool

检查是否满足收敛条件。

# 条件
1. 最近几次误差都小于容差
2. 秩已稳定
3. 没有新增全局枢轴（可选）
4. 或者已达到最大键维度
"""
function convergencecriterion(
    ranks::AbstractVector{Int},
    errors::AbstractVector{Float64},
    nglobalpivots::AbstractVector{Int},
    tolerance::Float64,
    maxbonddim::Int,
    ncheckhistory::Int;
    checkconvglobalpivot::Bool=true
)::Bool
    if length(errors) < ncheckhistory
        return false
    end
    lastranks = last(ranks, ncheckhistory)
    lastngpivots = last(nglobalpivots, ncheckhistory)
    return (
        all(last(errors, ncheckhistory) .< tolerance) &&
        (checkconvglobalpivot ? all(lastngpivots .== 0) : true) &&
        minimum(lastranks) == lastranks[end]
    ) || all(lastranks .>= maxbonddim)
end

"""
    GlobalPivotSearchInput(tci::TensorCI2{ValueType}) where {ValueType}

从TensorCI2对象构造GlobalPivotSearchInput。
"""
function GlobalPivotSearchInput(tci::TensorCI2{ValueType}) where {ValueType}
    return GlobalPivotSearchInput{ValueType}(
        tci.localdims,
        TensorTrain(tci),
        tci.maxsamplevalue,
        tci.Iset,
        tci.Jset
    )
end

# ===================================================================
# 主优化函数
# ===================================================================

"""
    optimize!(tci::TensorCI2{ValueType}, f; kwargs...) where {ValueType}

对TCI2执行优化扫描。

# 参数
- `tci::TensorCI2{ValueType}`: 要优化的TCI
- `f`: 目标函数
- `tolerance::Float64`: 目标容差（默认 1e-8）
- `maxbonddim::Int`: 最大键维度
- `maxiter::Int`: 最大迭代次数
- `sweepstrategy::Symbol`: 扫描策略
- `pivotsearch::Symbol`: 枢轴搜索策略 (:full 或 :rook)
- `verbosity::Int`: 输出详细程度
- `normalizeerror::Bool`: 是否归一化误差
- `globalpivotfinder`: 全局枢轴搜索器
- `maxnglobalpivot::Int`: 每次最多添加的全局枢轴数

# 返回值
- `ranks::Vector{Int}`: 每次迭代的秩
- `errors::Vector{Float64}`: 每次迭代的误差

# 算法
1. 执行2-site扫描更新索引集
2. 搜索全局枢轴
3. 重复直到收敛

# 提示
- 设置 tolerance > 0 或合理的 maxbonddim，否则无法收敛
- 使用 CachedFunction 包装昂贵的函数
"""
function optimize!(
    tci::TensorCI2{ValueType},
    f;
    tolerance::Union{Float64, Nothing}=nothing,
    pivottolerance::Union{Float64, Nothing}=nothing,
    maxbonddim::Int=typemax(Int),
    maxiter::Int=20,
    sweepstrategy::Symbol=:backandforth,
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    loginterval::Int=10,
    normalizeerror::Bool=true,
    ncheckhistory::Int=3,
    globalpivotfinder::Union{AbstractGlobalPivotFinder, Nothing}=nothing,
    maxnglobalpivot::Int=5,
    nsearchglobalpivot::Int=5,
    tolmarginglobalsearch::Float64=10.0,
    strictlynested::Bool=false,
    checkbatchevaluatable::Bool=false,
    checkconvglobalpivot::Bool=true
) where {ValueType}
    errors = Float64[]
    ranks = Int[]
    nglobalpivots = Int[]
    local tol::Float64

    if checkbatchevaluatable && !(f isa BatchEvaluator)
        error("Function `f` is not batch evaluatable")
    end

    if nsearchglobalpivot > 0 && nsearchglobalpivot < maxnglobalpivot
        error("nsearchglobalpivot < maxnglobalpivot!")
    end

    # 处理废弃的pivottolerance选项
    if !isnothing(pivottolerance)
        if !isnothing(tolerance) && (tolerance != pivottolerance)
            throw(ArgumentError("Got different values for pivottolerance and tolerance in optimize!(TCI2). For TCI2, both of these options have the same meaning. Please assign only `tolerance`."))
        else
            @warn "The option `pivottolerance` of `optimize!(tci::TensorCI2, f)` is deprecated. Please update your code to use `tolerance`, as `pivottolerance` will be removed in the future."
            tol = pivottolerance
        end
    elseif !isnothing(tolerance)
        tol = tolerance
    else
        tol = 1e-8  # 默认值
    end

    tstart = time_ns()

    if maxbonddim >= typemax(Int) && tol <= 0
        throw(ArgumentError(
            "Specify either tolerance > 0 or some maxbonddim; otherwise, the convergence criterion is not reachable!"
        ))
    end

    # 创建全局枢轴搜索器
    finder = if isnothing(globalpivotfinder)
        DefaultGlobalPivotFinder(
            nsearch=nsearchglobalpivot,
            maxnglobalpivot=maxnglobalpivot,
            tolmarginglobalsearch=tolmarginglobalsearch
        )
    else
        globalpivotfinder
    end

    globalpivots = MultiIndex[]
    
    # 主迭代循环
    for iter in 1:maxiter
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        abstol = tol * errornormalization;

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: starting 2site sweep")
            flush(stdout)
        end

        # 执行2-site扫描
        sweep2site!(
            tci, f, 2;
            iter1 = 1,
            abstol=abstol,
            maxbonddim=maxbonddim,
            pivotsearch=pivotsearch,
            strictlynested=strictlynested,
            verbosity=verbosity,
            sweepstrategy=sweepstrategy,
            fillsitetensors=true
        )
        
        if verbosity > 0 && length(globalpivots) > 0 && mod(iter, loginterval) == 0
            abserr = [abs(evaluate(tci, p) - f(p)) for p in globalpivots]
            nrejections = length(abserr .> abstol)
            if nrejections > 0
                println("  Rejected $(nrejections) global pivots added in the previous iteration, errors are $(abserr)")
                flush(stdout)
            end
        end
        push!(errors, pivoterror(tci))

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: start searching global pivots")
            flush(stdout)
        end

        # 搜索全局枢轴
        input = GlobalPivotSearchInput(tci)
        globalpivots = finder(
            input, f, abstol;
            verbosity=verbosity,
            rng=Random.default_rng()
        )
        addglobalpivots!(tci, globalpivots)
        push!(nglobalpivots, length(globalpivots))

        if verbosity > 1
            println("  Walltime $(1e-9*(time_ns() - tstart)) sec: done searching global pivots")
            flush(stdout)
        end

        push!(ranks, rank(tci))
        if verbosity > 0 && mod(iter, loginterval) == 0
            println("iteration = $iter, rank = $(last(ranks)), error= $(last(errors)), maxsamplevalue= $(tci.maxsamplevalue), nglobalpivot=$(length(globalpivots))")
            flush(stdout)
        end
        
        # 检查收敛
        if convergencecriterion(
            ranks, errors,
            nglobalpivots,
            abstol, maxbonddim, ncheckhistory;
            checkconvglobalpivot=checkconvglobalpivot
        )
            break
        end
    end

    # 最后执行1-site扫描以：
    # (1) 移除全局枢轴添加的不必要枢轴
    # (2) 计算站点张量
    errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
    abstol = tol * errornormalization;
    sweep1site!(
        tci,
        f,
        abstol=abstol,
        maxbonddim=maxbonddim,
    )

    _sanitycheck(tci)

    return ranks, errors ./ errornormalization
end

# ===================================================================
# 2-site扫描
# ===================================================================

"""
    sweep2site!(tci::TensorCI2{ValueType}, f, niter::Int; kwargs...) where {ValueType}

执行2-site扫描。

# 参数
- `niter`: 迭代次数
- `abstol`: 绝对容差
- `maxbonddim`: 最大键维度
- `sweepstrategy`: 扫描策略
- `pivotsearch`: 枢轴搜索策略
- `strictlynested`: 是否严格嵌套
- `fillsitetensors`: 是否填充站点张量

# 说明
2-site扫描同时更新两个相邻站点，可以更好地优化键维度。
"""
function sweep2site!(
    tci::TensorCI2{ValueType}, f, niter::Int;
    iter1::Int=1,
    abstol::Float64=1e-8,
    maxbonddim::Int=typemax(Int),
    sweepstrategy::Symbol=:backandforth,
    pivotsearch::Symbol=:full,
    verbosity::Int=0,
    strictlynested::Bool=false,
    fillsitetensors::Bool=true
) where {ValueType}
    invalidatesitetensors!(tci)

    n = length(tci)

    for iter in iter1:iter1+niter-1
        # 从历史获取额外索引（非严格嵌套模式）
        extraIset = [MultiIndex[] for _ in 1:n]
        extraJset = [MultiIndex[] for _ in 1:n]
        if !strictlynested && length(tci.Iset_history) > 0
            extraIset = tci.Iset_history[end]
            extraJset = tci.Jset_history[end]
        end

        # 保存当前索引集到历史
        push!(tci.Iset_history, deepcopy(tci.Iset))
        push!(tci.Jset_history, deepcopy(tci.Jset))

        flushpivoterror!(tci)
        
        if forwardsweep(sweepstrategy, iter) # 正向扫描
            for bondindex in 1:n-1
                updatepivots!(
                    tci, bondindex, f, true;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    sweepdirection=:forward,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    extraIset=extraIset[bondindex+1],
                    extraJset=extraJset[bondindex],
                )
            end
        else # 反向扫描
            for bondindex in (n-1):-1:1
                updatepivots!(
                    tci, bondindex, f, false;
                    abstol=abstol,
                    maxbonddim=maxbonddim,
                    sweepdirection=:backward,
                    pivotsearch=pivotsearch,
                    verbosity=verbosity,
                    extraIset=extraIset[bondindex+1],
                    extraJset=extraJset[bondindex],
                )
            end
        end
    end

    if fillsitetensors
        fillsitetensors!(tci, f)
    end
    nothing
end

# ===================================================================
# 主函数：crossinterpolate2
# ===================================================================

@doc raw"""
    crossinterpolate2(::Type{ValueType}, f, localdims, initialpivots; kwargs...) where {ValueType}

使用TCI2算法对函数进行交叉插值。

# 数学背景
对函数 f(u)（其中 u ∈ [1,...,d₁] × [1,...,d₂] × ... × [1,...,dₙ]）
进行张量列近似。TCI2是推荐的默认算法。

# 参数
- `ValueType`: f 的返回类型
- `f`: 要插值的函数
- `localdims`: 各维度大小
- `initialpivots`: 初始枢轴列表（默认 [[1,1,...]]）

# 关键字参数
参见 [`optimize!`](@ref) 获取完整的关键字参数列表，包括：
- `tolerance`, `maxbonddim`, `maxiter` 等

# 返回值
- `tci::TensorCI2{ValueType}`: TCI2对象
- `ranks::Vector{Int}`: 每次迭代的秩
- `errors::Vector{Float64}`: 每次迭代的误差

# 提示
- 设置 tolerance > 0 或合理的 maxbonddim，否则无法收敛
- 使用 CachedFunction 包装昂贵的函数

# 示例
```julia
# 定义函数
f(x) = exp(-sum(x.^2))

# 进行交叉插值
tci, ranks, errors = crossinterpolate2(Float64, f, [10, 10, 10])

# 转换为张量列
tt = tensortrain(tci)

# 求值
println(tt([5, 5, 5]))
```

# 另见
- [`optimize!`](@ref): TCI2优化函数
- [`optfirstpivot`](@ref): 优化第一个枢轴
- [`CachedFunction`](@ref): 函数缓存
- [`crossinterpolate1`](@ref): TCI1算法
"""
function crossinterpolate2(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    initialpivots::Vector{MultiIndex}=[ones(Int, length(localdims))];
    kwargs...
) where {ValueType,N}
    tci = TensorCI2{ValueType}(f, localdims, initialpivots)
    ranks, errors = optimize!(tci, f; kwargs...)
    return tci, ranks, errors
end

# ===================================================================
# 全局枢轴搜索
# ===================================================================

"""
    searchglobalpivots(tci::TensorCI2{ValueType}, f, abstol; kwargs...) where {ValueType}

搜索插值误差超过 abstol 的全局枢轴。

# 参数
- `tci`: TensorCI2对象
- `f`: 原始函数
- `abstol`: 绝对容差
- `nsearch`: 搜索点数量
- `maxnglobalpivot`: 最多返回的枢轴数

# 返回值
- 误差超过阈值的枢轴列表
"""
function searchglobalpivots(
    tci::TensorCI2{ValueType}, f, abstol;
    verbosity::Int=0,
    nsearch::Int = 100,
    maxnglobalpivot::Int = 5
)::Vector{MultiIndex} where {ValueType}
    if nsearch == 0 || maxnglobalpivot == 0
        return MultiIndex[]
    end

    if !issitetensorsavailable(tci)
        fillsitetensors!(tci, f)
    end

    pivots = Dict{Float64,MultiIndex}()
    ttcache = TTCache(tci)
    
    for _ in 1:nsearch
        pivot, error = _floatingzone(ttcache, f; earlystoptol = 10 * abstol, nsweeps=100)
        if error > abstol
            pivots[error] = pivot
        end
        if length(pivots) == maxnglobalpivot
            break
        end
    end

    if length(pivots) == 0
        if verbosity > 1
            println("  No global pivot found")
        end
        return MultiIndex[]
    end

    if verbosity > 1
        maxerr = maximum(keys(pivots))
        println("  Found $(length(pivots)) global pivots: max error $(maxerr)")
    end

    return [p for (_,p) in pivots]
end
