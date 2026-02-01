# ===================================================================
# tensorci1.jl - TCI1算法实现
# ===================================================================
# 这个文件实现了TensorCI1类型和crossinterpolate1函数。
#
# TCI1（Tensor Cross Interpolation 1）是张量交叉插值的第一种算法，
# 由Oseledets等人提出。它使用嵌套索引集(nested index sets)来
# 构建张量列近似。
#
# 主要特点：
# - 存储完整的Pi矩阵，支持高效的增量更新
# - 使用ACA算法来选择最优枢轴
# - 支持正向和反向扫描优化
# ===================================================================

"""
    mutable struct TensorCI1{ValueType} <: AbstractTensorTrain{ValueType}

TCI1算法产生的张量交叉插值表示。

# 背景
TCI1是Oseledets提出的原始TCI算法。它构建如下近似：
```math
f(i_1, ..., i_n) ≈ ∏_{k=1}^{n} T_k[i_k] * P_k^{-1}
```
其中 T_k 是3维张量，P_k 是枢轴矩阵。

# 类型参数
- `ValueType`: 张量元素的类型

# 字段
- `Iset::Vector{IndexSet{MultiIndex}}`: 每个位置的左索引集
  - Iset[p] 包含位置 p 的所有左复合索引
- `Jset::Vector{IndexSet{MultiIndex}}`: 每个位置的右索引集
  - Jset[p] 包含位置 p 的所有右复合索引
- `localdims::Vector{Int}`: 各维度的大小
- `T::Vector{Array{ValueType,3}}`: T张量列表
  - T[p] 是位置 p 的3维张量
  - 形状：(|Iset[p]|, localdims[p], |Jset[p]|)
- `P::Vector{Matrix{ValueType}}`: P矩阵列表
  - P[p] 是位置 p 的枢轴矩阵
- `aca::Vector{MatrixACA{ValueType}}`: ACA对象列表
  - 用于高效的枢轴选择和增量更新
- `Pi::Vector{Matrix{ValueType}}`: Π矩阵列表
  - Pi[p] 存储完整的4腿张量（展平为2D）
- `PiIset::Vector{IndexSet{MultiIndex}}`: Π矩阵的左索引集
- `PiJset::Vector{IndexSet{MultiIndex}}`: Π矩阵的右索引集
- `pivoterrors::Vector{Float64}`: 每个键的枢轴误差
- `maxsamplevalue::Float64`: 函数的最大采样值

# 与TCI2的区别
- TCI1存储完整的Pi矩阵，支持增量更新
- TCI2只存储站点张量，更节省内存
- TCI1使用ACA进行枢轴选择
- TCI2使用LU分解进行枢轴选择

# 创建方式
通常通过 [`crossinterpolate1`](@ref) 函数创建：
```julia
tci, ranks, errors = crossinterpolate1(Float64, f, [10, 10, 10])
```

# 另见
- [`crossinterpolate1`](@ref): TCI1算法的主函数
- [`TensorCI2`](@ref): TCI2算法的表示
"""
mutable struct TensorCI1{ValueType} <: AbstractTensorTrain{ValueType}
    Iset::Vector{IndexSet{MultiIndex}}   # 左索引集
    Jset::Vector{IndexSet{MultiIndex}}   # 右索引集
    localdims::Vector{Int}               # 局部维度

    """
    T张量：来自TCI论文的3腿张量。
    第一和第三腿是连接相邻 P⁻¹ 矩阵的键，第二腿是局部索引。
    T[p] 是第 p 个站点张量。
    """
    T::Vector{Array{ValueType,3}}

    """
    P张量：来自TCI论文的2腿枢轴矩阵。
    """
    P::Vector{Matrix{ValueType}}

    """
    ACA对象：随着新枢轴的添加而更新。
    用于高效地选择最优枢轴和增量更新Pi矩阵。
    """
    aca::Vector{MatrixACA{ValueType}}

    """
    Π矩阵：来自TCI论文的4腿张量。
    通过分解得到 T 和 P。
    保存在内存中以支持高效更新。
    """
    Pi::Vector{Matrix{ValueType}}
    PiIset::Vector{IndexSet{MultiIndex}}  # Π的左索引集
    PiJset::Vector{IndexSet{MultiIndex}}  # Π的右索引集

    """
    每个站点的枢轴误差。
    """
    pivoterrors::Vector{Float64}
    maxsamplevalue::Float64

    """
        TensorCI1{ValueType}(localdims::AbstractVector{Int}) where {ValueType}
    
    创建一个空的TensorCI1对象。
    
    # 参数
    - `localdims`: 各维度的大小
    """
    function TensorCI1{ValueType}(
        localdims::AbstractVector{Int}
    ) where {ValueType}
        n = length(localdims)
        new{ValueType}(
            [IndexSet{MultiIndex}() for _ in 1:n],  # Iset（空索引集）
            [IndexSet{MultiIndex}() for _ in 1:n],  # Jset（空索引集）
            collect(localdims),                     # localdims
            [zeros(0, d, 0) for d in localdims],    # T（空张量）
            [zeros(0, 0) for _ in 1:n],             # P（空矩阵）
            [MatrixACA(ValueType, 0, 0) for _ in 1:n],  # aca
            [zeros(0, 0) for _ in 1:n],             # Pi（空矩阵）
            [IndexSet{MultiIndex}() for _ in 1:n],  # PiIset
            [IndexSet{MultiIndex}() for _ in 1:n],  # PiJset
            fill(Inf, n - 1),                       # pivoterrors（初始化为无穷大）
            0.0                                     # maxsamplevalue
        )
    end
end

"""
    TensorCI1{ValueType}(localdims::NTuple{N,Int}) where {ValueType,N}

从元组创建TensorCI1。
"""
function TensorCI1{ValueType}(
    localdims::NTuple{N,Int},
) where {ValueType,N}
    return TensorCI1{ValueType}(collect(localdims)::Vector{Int})
end

"""
    TensorCI1{ValueType}(localdim::Int, length::Int) where {ValueType}

创建所有维度相同的TensorCI1。
"""
function TensorCI1{ValueType}(
    localdim::Int,
    length::Int
) where {ValueType}
    return TensorCI1{ValueType}(fill(localdim, length))
end

"""
    TensorCI1{ValueType}(func, localdims, firstpivot) where {ValueType}

从函数和第一个枢轴创建TensorCI1。

# 参数
- `func`: 要插值的函数
- `localdims`: 各维度大小
- `firstpivot`: 第一个枢轴（必须满足 f(firstpivot) ≠ 0）

# 算法
1. 初始化索引集为第一个枢轴的分割
2. 构建所有Pi矩阵
3. 从每个Pi矩阵提取T和P
"""
function TensorCI1{ValueType}(
    func::F,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    firstpivot::Vector{Int}
) where {F,ValueType,N}
    tci = TensorCI1{ValueType}(localdims)
    # 包装函数以确保类型稳定
    f = x -> convert(ValueType, func(x))

    # 验证第一个枢轴
    tci.maxsamplevalue = abs(f(firstpivot))
    if tci.maxsamplevalue == 0
        throw(ArgumentError("Please provide a first pivot where f(pivot) != 0."))
    end
    if length(localdims) != length(firstpivot)
        throw(ArgumentError("Firstpivot and localdims must have same length."))
    end

    n = length(localdims)
    
    # 初始化索引集为第一个枢轴的分割
    # Iset[p] 包含第一个枢轴的前 p-1 个分量
    tci.Iset = [IndexSet([firstpivot[1:p-1]]) for p in 1:n]
    # Jset[p] 包含第一个枢轴的后 n-p 个分量
    tci.Jset = [IndexSet([firstpivot[p+1:end]]) for p in 1:n]
    
    # 构建Pi索引集和Pi矩阵
    tci.PiIset = [getPiIset(tci, p) for p in 1:n]
    tci.PiJset = [getPiJset(tci, p) for p in 1:n]
    tci.Pi = [getPi(tci, p, f) for p in 1:n-1]

    # 从每个Pi矩阵提取T和P
    for p in 1:n-1
        # 计算局部枢轴位置
        localpivot = (
            pos(tci.PiIset[p], tci.Iset[p+1][1]),     # 在Pi左索引中的位置
            pos(tci.PiJset[p+1], tci.Jset[p][1]))      # 在Pi右索引中的位置
        
        # 创建ACA对象
        tci.aca[p] = MatrixACA(tci.Pi[p], localpivot)
        
        # 更新T张量
        if p == 1
            updateT!(tci, 1, tci.Pi[p][:, [localpivot[2]]])
        end
        updateT!(tci, p + 1, tci.Pi[p][[localpivot[2]], :])
        
        # 设置P矩阵（枢轴值）
        tci.P[p] = tci.Pi[p][[localpivot[1]], [localpivot[2]]]
    end
    tci.P[end] = ones(ValueType, 1, 1)  # 最后一个P是1×1单位矩阵

    return tci
end

"""
    Base.broadcastable(tci::TensorCI1{V}) where {V}

使TensorCI1支持广播。
"""
function Base.broadcastable(tci::TensorCI1{V}) where {V}
    return Ref(tci)
end

"""
    lastsweeppivoterror(tci::TensorCI1{V}) where {V}

获取上一次扫描的最大枢轴误差。
"""
function lastsweeppivoterror(tci::TensorCI1{V}) where {V}
    return maximum(tci.pivoterrors)
end

"""
    updatemaxsample!(tci::TensorCI1{V}, samples::Array{V}) where {V}

更新最大采样值。
"""
function updatemaxsample!(tci::TensorCI1{V}, samples::Array{V}) where {V}
    tci.maxsamplevalue = maxabs(tci.maxsamplevalue, samples)
end

# ===================================================================
# 站点张量计算
# ===================================================================

"""
    TtimesPinv(tci::TensorCI1{V}, p::Int) where {V}

计算 T[p] * P[p]⁻¹。

# 说明
这是TCI1的站点张量公式。
使用数值稳定的 AtimesBinv 函数。
"""
function TtimesPinv(tci::TensorCI1{V}, p::Int) where {V}
    T = tci.T[p]
    shape = size(T)
    # 将T重塑为2D，乘以P逆，再重塑回3D
    TPinv = AtimesBinv(reshape(T, (shape[1] * shape[2], shape[3])), tci.P[p])
    return reshape(TPinv, shape)
end

"""
    PinvtimesT(tci::TensorCI1{V}, p::Int) where {V}

计算 P[p-1]⁻¹ * T[p]。
"""
function PinvtimesT(tci::TensorCI1{V}, p::Int) where {V}
    T = tci.T[p]
    shape = size(T)
    PinvT = AinvtimesB(tci.P[p-1], reshape(T, shape[1], shape[2] * shape[3]))
    return reshape(PinvT, shape)
end

"""
    sitetensor(tci::TensorCI1{V}, p::Int) where {V}

获取位置 p 的站点张量。
"""
function sitetensor(tci::TensorCI1{V}, p::Int) where {V}
    return TtimesPinv(tci, p)
end

function sitetensor(tci::TensorCI1{V}, p) where {V}
    return sitetensor.(tci, p)
end

"""
    sitetensors(tci::TensorCI1{V}) where {V}

获取所有站点张量的数组。
"""
function sitetensors(tci::TensorCI1{V}) where {V}
    return [sitetensor(tci, p) for p in 1:length(tci.T)]
end

# ===================================================================
# 维度相关函数
# ===================================================================

"""
    length(tci::TensorCI1{V}) where {V}

获取TCI1的长度（站点数量）。
"""
function length(tci::TensorCI1{V}) where {V}
    return length(tci.T)
end

"""
    linkdims(tci::TensorCI1{V}) where {V}

获取所有键维度。
"""
function linkdims(tci::TensorCI1{V}) where {V}
    return [size(T, 1) for T in tci.T[2:end]]
end

"""
    linkdim(tci::TensorCI1{V}, i::Int) where {V}

获取第 i 个键的维度。
"""
function linkdim(tci::TensorCI1{V}, i::Int) where {V}
    return size(tci.T[i+1], 1)
end

"""
    sitedims(tci::TensorCI1{V}) where {V}

获取所有站点维度。
"""
function sitedims(tci::TensorCI1{V}) where {V}
    return [collect(size(T)[2:end-1]) for T in tci.T]
end

"""
    sitedim(tci::TensorCI1{V}, i::Int) where {V}

获取第 i 个站点的维度。
"""
function sitedim(tci::TensorCI1{V}, i::Int) where {V}
    return collect(size(tci.T[i])[2:end-1])
end

# ===================================================================
# 求值函数
# ===================================================================

"""
    evaluate(tci::TensorCI1{V}, indexset) where {V}

在指定索引处求值TCI1。

# 警告
此方法效率较低。如需多次求值，请先转换为TensorTrain：
```julia
tt = tensortrain(tci)
value = tt(indexset)
```
"""
function evaluate(
    tci::TensorCI1{V},
    indexset::Union{AbstractVector{LocalIndex},NTuple{N,LocalIndex}}
)::V where {N,V}
    # 计算所有 T[p][:,indexset[p],:] * P[p]⁻¹ 的乘积
    return prod(
        AtimesBinv(tci.T[p][:, indexset[p], :], tci.P[p])
        for p in 1:length(tci))[1, 1]
end

# ===================================================================
# Pi矩阵相关函数
# ===================================================================

"""
    getPiIset(tci::TensorCI1{V}, p::Int) where {V}

获取位置 p 的Π矩阵的左索引集。

# 说明
Pi的左索引是 Iset[p] 与 1:localdims[p] 的笛卡尔积。
"""
function getPiIset(tci::TensorCI1{V}, p::Int) where {V}
    return IndexSet([
        [is..., ups] for is in tci.Iset[p].fromint, ups in 1:tci.localdims[p]
    ][:])
end

"""
    getPiJset(tci::TensorCI1{V}, p::Int) where {V}

获取位置 p 的Π矩阵的右索引集。

# 说明
Pi的右索引是 1:localdims[p] 与 Jset[p] 的笛卡尔积。
"""
function getPiJset(tci::TensorCI1{V}, p::Int) where {V}
    return IndexSet([
        [up1s, js...] for up1s in 1:tci.localdims[p], js in tci.Jset[p].fromint
    ][:])
end

"""
    getPi(tci::TensorCI1{V}, p::Int, f) where {V}

构建位置 p 的Π矩阵。

# 数学说明
Π[p] 的元素 (i, j) 是：
Π[is, up, up+1, js] = f([is..., up, up+1, js...])

在TCI论文中，索引顺序是 (i, u_p, u_{p+1}, j)。
"""
function getPi(tci::TensorCI1{V}, p::Int, f::F) where {V,F}
    iset = tci.PiIset[p]
    jset = tci.PiJset[p+1]
    # 对所有索引组合求值
    res = [f([is..., js...]) for is in iset.fromint, js in jset.fromint]
    updatemaxsample!(tci, res)
    return res
end

"""
    getcross(tci::TensorCI1{V}, p::Int) where {V}

获取位置 p 的MatrixCI对象。

# 说明
将Pi矩阵封装为MatrixCI，用于枢轴选择。
"""
function getcross(tci::TensorCI1{V}, p::Int) where {V}
    # 将索引从相邻站点的顺序转换为当前站点的顺序
    iset = pos(tci.PiIset[p], tci.Iset[p+1].fromint)
    jset = pos(tci.PiJset[p+1], tci.Jset[p].fromint)
    
    # 重塑T张量
    shape = size(tci.T[p])
    Tp = reshape(tci.T[p], (shape[1] * shape[2], shape[3]))
    shape1 = size(tci.T[p+1])
    Tp1 = reshape(tci.T[p+1], (shape1[1], shape1[2] * shape1[3]))
    
    return MatrixCI(iset, jset, Tp, Tp1)
end

# ===================================================================
# T张量更新
# ===================================================================

"""
    updateT!(tci::TensorCI1{V}, p::Int, new_T::AbstractArray{V}) where {V}

更新位置 p 的T张量。

# 参数
- `tci`: TensorCI1对象
- `p`: 位置索引
- `new_T`: 新的T数据（会被重塑）
"""
function updateT!(
    tci::TensorCI1{V},
    p::Int,
    new_T::AbstractArray{V}
) where {V}
    tci.T[p] = reshape(
        new_T,
        length(tci.Iset[p]),
        tci.localdims[p],
        length(tci.Jset[p]))
end

# ===================================================================
# Pi矩阵增量更新
# ===================================================================

"""
    updatePirows!(tci::TensorCI1{V}, p::Int, f) where {V}

添加新行后更新Pi矩阵。

# 说明
当Iset扩展时，Pi矩阵需要添加新行。
这个函数高效地只计算新增的行。
"""
function updatePirows!(tci::TensorCI1{V}, p::Int, f::F) where {V,F}
    newIset = getPiIset(tci, p)
    # 找出新增的索引
    diffIset = setdiff(newIset.fromint, tci.PiIset[p].fromint)
    
    # 创建新的Pi矩阵
    newPi = Matrix{V}(undef, length(newIset), size(tci.Pi[p], 2))

    # 计算旧索引在新Pi中的位置
    permutation = [pos(newIset, imulti) for imulti in tci.PiIset[p].fromint]
    newPi[permutation, :] = tci.Pi[p]  # 复制旧数据

    # 计算新增的行
    for imulti in diffIset
        newi = pos(newIset, imulti)
        row = [f([imulti..., js...]) for js in tci.PiJset[p+1].fromint]
        newPi[newi, :] = row
        updatemaxsample!(tci, row)
    end
    
    tci.Pi[p] = newPi
    tci.PiIset[p] = newIset

    # 更新ACA对象
    Tshape = size(tci.T[p])
    Tp = reshape(tci.T[p], (Tshape[1] * Tshape[2], Tshape[3]))
    setrows!(tci.aca[p], Tp, permutation)
end

"""
    updatePicols!(tci::TensorCI1{V}, p::Int, f) where {V}

添加新列后更新Pi矩阵。

# 说明
当Jset扩展时，Pi矩阵需要添加新列。
"""
function updatePicols!(tci::TensorCI1{V}, p::Int, f::F) where {V,F}
    newJset = getPiJset(tci, p + 1)
    diffJset = setdiff(newJset.fromint, tci.PiJset[p+1].fromint)
    
    newPi = Matrix{V}(undef, size(tci.Pi[p], 1), length(newJset))

    permutation = [pos(newJset, jmulti) for jmulti in tci.PiJset[p+1].fromint]
    newPi[:, permutation] = tci.Pi[p]

    for jmulti in diffJset
        newj = pos(newJset, jmulti)
        col = [f([is..., jmulti...]) for is in tci.PiIset[p].fromint]
        newPi[:, newj] = col
        updatemaxsample!(tci, col)
    end

    tci.Pi[p] = newPi
    tci.PiJset[p+1] = newJset

    Tshape = size(tci.T[p+1])
    Tp = reshape(tci.T[p+1], (Tshape[1], Tshape[2] * Tshape[3]))
    setcols!(tci.aca[p], Tp, permutation)
end

# ===================================================================
# 枢轴添加函数
# ===================================================================

"""
    addpivotrow!(tci::TensorCI1{V}, cross::MatrixCI{V}, p::Int, newi::Int, f) where {V}

将索引 newi 的行添加为键 p 的枢轴行。

# 参数
- `tci`: TensorCI1对象
- `cross`: 对应的MatrixCI对象
- `p`: 键索引
- `newi`: 新枢轴行的索引
- `f`: 原始函数
"""
function addpivotrow!(tci::TensorCI1{V}, cross::MatrixCI{V}, p::Int, newi::Int, f) where {V}
    # 更新ACA和交叉对象
    addpivotrow!(tci.aca[p], tci.Pi[p], newi)
    addpivotrow!(cross, tci.Pi[p], newi)
    
    # 将新索引添加到Iset
    push!(tci.Iset[p+1], tci.PiIset[p][newi])
    
    # 更新T和P
    updateT!(tci, p + 1, cross.pivotrows)
    tci.P[p] = pivotmatrix(cross)

    # 更新相邻的Pi矩阵（因为共享的T已改变）
    if p < length(tci) - 1
        updatePirows!(tci, p + 1, f)
    end
end

"""
    addpivotcol!(tci::TensorCI1{V}, cross::MatrixCI{V}, p::Int, newj::Int, f) where {V}

将索引 newj 的列添加为键 p 的枢轴列。
"""
function addpivotcol!(tci::TensorCI1{V}, cross::MatrixCI{V}, p::Int, newj::Int, f) where {V}
    addpivotcol!(tci.aca[p], tci.Pi[p], newj)
    addpivotcol!(cross, tci.Pi[p], newj)
    
    push!(tci.Jset[p], tci.PiJset[p+1][newj])
    
    updateT!(tci, p, cross.pivotcols)
    tci.P[p] = pivotmatrix(cross)

    # 更新相邻的Pi矩阵
    if p > 1
        updatePicols!(tci, p - 1, f)
    end
end

"""
    addpivot!(tci::TensorCI1{V}, p::Int, f, tolerance=1e-12) where {V}

在位置 p 添加一个枢轴。

# 参数
- `tci`: TensorCI1对象
- `p`: 键位置
- `f`: 原始函数
- `tolerance`: 误差阈值（误差低于此值时不添加枢轴）

# 算法
1. 使用ACA找到误差最大的位置
2. 如果误差大于容差，添加该枢轴
"""
function addpivot!(tci::TensorCI1{V}, p::Int, f::F, tolerance::Float64=1e-12) where {V,F}
    # 验证位置
    if (p < 1) || (p > length(tci) - 1)
        throw(BoundsError(
            "Pi tensors can only be built at sites 1 to length - 1 = $(length(tci) - 1)."))
    end

    # 检查是否已达到满秩
    if rank(tci.aca[p]) >= minimum(size(tci.Pi[p]))
        tci.pivoterrors[p] = 0.0
        return
    end

    # 找到新枢轴
    newpivot, newerror = findnewpivot(tci.aca[p], tci.Pi[p])

    tci.pivoterrors[p] = newerror
    
    # 检查误差是否足够小
    if newerror < tolerance
        return
    end

    # 添加枢轴
    cross = getcross(tci, p)
    addpivotcol!(tci, cross, p, newpivot[2], f)
    addpivotrow!(tci, cross, p, newpivot[1], f)
end

# ===================================================================
# 全局枢轴相关函数
# ===================================================================

"""
    crosserror(tci::TensorCI1{T}, f, x::MultiIndex, y::MultiIndex)::Float64 where {T}

计算交叉插值在 (x, y) 处的误差。

# 说明
误差 = |f([x..., y...]) - TCI([x..., y...])|
"""
function crosserror(tci::TensorCI1{T}, f, x::MultiIndex, y::MultiIndex)::Float64 where {T}
    if isempty(x) || isempty(y)
        return 0.0
    end

    bondindex = length(x)
    
    # 如果索引已在集合中，误差为0
    if x in tci.Iset[bondindex+1] || y in tci.Jset[bondindex]
        return 0.0
    end

    if (isempty(tci.Jset[bondindex]))
        return abs(f(vcat(x, y)))
    end

    # 计算左半部分和右半部分
    fx = [f(vcat(x, j)) for j in tci.Jset[bondindex]]
    fy = [f(vcat(i, y)) for i in tci.Iset[bondindex+1]]
    updatemaxsample!(tci, fx)
    updatemaxsample!(tci, fy)
    
    # 计算交叉插值值并与真实值比较
    return abs(first(AtimesBinv(transpose(fx), tci.P[bondindex]) * fy) - f(vcat(x, y)))
end

"""
    updateIproposal(tci, f, newpivot, newI, newJ, abstol)

更新I提案（内部函数）。
"""
function updateIproposal(
    tci::TensorCI1{T},
    f,
    newpivot::Vector{Int},
    newI::Vector{MultiIndex},
    newJ::Vector{MultiIndex},
    abstol::Float64
) where {T}
    error = Inf
    for bondindex in 1:length(tci)-1
        if isempty(newI[bondindex+1])
            error = 0.0
            continue
        end

        if error > abstol
            newI[bondindex+1] = vcat(newI[bondindex], newpivot[bondindex])
            error = crosserror(tci, f, newI[bondindex+1], newJ[bondindex])
        elseif newpivot[1:bondindex] in tci.Iset[bondindex]
            newI[bondindex+1] = newpivot[1:bondindex+1]
            error = crosserror(tci, f, newI[bondindex+1], newJ[bondindex])
        else
            xset = [vcat(i, newpivot[bondindex]) for i in tci.Iset[bondindex]]
            errors = [crosserror(tci, f, x, newJ[bondindex]) for x in xset]
            maxindex = argmax(errors)
            newI[bondindex+1] = xset[maxindex]
            error = errors[maxindex]
        end

        if error < abstol
            newI[bondindex+1] = MultiIndex()
        end
    end
    return newI
end

"""
    updateJproposal(tci, f, newpivot, newI, newJ, abstol)

更新J提案（内部函数）。
"""
function updateJproposal(
    tci::TensorCI1{T},
    f,
    newpivot::Vector{Int},
    newI::Vector{MultiIndex},
    newJ::Vector{MultiIndex},
    abstol::Float64
) where {T}
    error = Inf
    for bondindex in length(tci)-1:-1:1
        if isempty(newJ[bondindex])
            error = 0.0
            continue
        end

        if error > abstol
            newJ[bondindex] = vcat(newpivot[bondindex+1], newJ[bondindex+1])
            error = crosserror(tci, f, newI[bondindex+1], newJ[bondindex])
        elseif newpivot[bondindex+2:end] in tci.Jset[bondindex+1]
            newJ[bondindex] = newpivot[bondindex+1:end]
            error = crosserror(tci, f, newI[bondindex+1], newJ[bondindex])
        else
            yset = [vcat(newpivot[bondindex+1], j) for j in tci.Jset[bondindex+1]]
            errors = [crosserror(tci, f, newI[bondindex+1], y) for y in yset]
            maxindex = argmax(errors)
            newJ[bondindex] = yset[maxindex]
            error = errors[maxindex]
        end

        if error < abstol
            newJ[bondindex] = MultiIndex()
        end
    end
    return newJ
end

"""
    addglobalpivot!(tci::TensorCI1{T}, f, newpivot::Vector{Int}, abstol::Float64) where {T}

添加一个全局枢轴。

# 参数
- `tci`: TensorCI1对象
- `f`: 原始函数
- `newpivot`: 完整的枢轴索引
- `abstol`: 绝对容差

# 说明
全局枢轴是一个完整的多索引，指定了在索引空间中的一个特定点。
这与单键枢轴不同，单键枢轴只指定一个键的行或列。
"""
function addglobalpivot!(
    tci::TensorCI1{T},
    f,
    newpivot::Vector{Int},
    abstol::Float64
) where {T}
    # 验证长度
    if length(newpivot) != length(tci)
        throw(DimensionMismatch(
            "New global pivot $newpivot should have the same length as the TCI, i.e.
            exactly $(length(tci)) entries."))
    end

    # 初始化I和J提案
    newI = [newpivot[1:p-1] for p in 1:length(tci)]
    newJ = [newpivot[p+1:end] for p in 1:length(tci)]
    newI = updateIproposal(tci, f, newpivot, newI, newJ, abstol)

    # 交替优化I和J
    for iter in 1:length(tci)
        newJ = updateJproposal(tci, f, newpivot, newI, newJ, abstol)
        newI = updateIproposal(tci, f, newpivot, newI, newJ, abstol)
        if isempty.(newI[2:end]) == isempty.(newJ[1:length(tci)-1])
            break
        end
    end

    # 添加枢轴行
    for p in 1:length(newI)-1
        if !isempty(newI[p+1])
            addpivotrow!(tci, getcross(tci, p), p, pos(tci.PiIset[p], newI[p+1]), f)
        end
    end

    # 添加枢轴列
    for p in length(newJ)-1:-1:1
        if !isempty(newJ[p])
            addpivotcol!(tci, getcross(tci, p), p, pos(tci.PiJset[p+1], newJ[p]), f)
        end
    end
end

# ===================================================================
# 主函数：crossinterpolate1
# ===================================================================

@doc raw"""
    crossinterpolate1(::Type{ValueType}, f, localdims, firstpivot; kwargs...) where {ValueType}

使用TCI1算法对函数进行交叉插值。

# 数学背景
对函数 f(u)（其中 u ∈ [1,...,d₁] × [1,...,d₂] × ... × [1,...,dₙ]）
进行张量列近似。

# 参数
- `ValueType`: f 的返回类型（如 Float64, ComplexF64）
- `f`: 要插值的函数。f 接受一个与 localdims 等长的向量参数
- `localdims`: 各维度大小的向量或元组
- `firstpivot`: 第一个枢轴，默认 [1, 1, ...]

# 关键字参数
- `tolerance::Float64=1e-8`: 收敛容差
- `maxiter::Int=200`: 最大扫描次数
- `sweepstrategy::Symbol=:backandforth`: 扫描策略
  - `:forward`: 只正向扫描
  - `:backward`: 只反向扫描
  - `:backandforth`: 交替正向和反向
- `pivottolerance::Float64=1e-12`: 添加枢轴的最小误差阈值
- `verbosity::Int=0`: 输出详细程度（>0 输出收敛信息）
- `additionalpivots::Vector{MultiIndex}=[]`: 额外的初始枢轴
- `normalizeerror::Bool=true`: 是否归一化误差

# 返回值
- `tci::TensorCI1{ValueType}`: TCI1对象
- `ranks::Vector{Int}`: 每次迭代的秩
- `errors::Vector{Float64}`: 每次迭代的误差

# 收敛
- 如果 `normalizeerror=true`：当相对误差 < tolerance 时停止
- 如果 `normalizeerror=false`：当绝对误差 < tolerance 时停止

# 提示
- 收敛可能依赖于第一个枢轴的选择。选择靠近 f 最大值的位置通常效果较好
- 使用 [`optfirstpivot`](@ref) 可以优化第一个枢轴

# 示例
```julia
# 定义一个简单函数
f(x) = exp(-sum(x.^2))

# 进行交叉插值
tci, ranks, errors = crossinterpolate1(Float64, f, [10, 10, 10])

# 转换为张量列并求值
tt = tensortrain(tci)
println(tt([5, 5, 5]))
```

# 另见
- [`optfirstpivot`](@ref): 优化第一个枢轴
- [`CachedFunction`](@ref): 函数缓存
- [`crossinterpolate2`](@ref): TCI2算法
"""
function crossinterpolate1(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    firstpivot::MultiIndex=ones(Int, length(localdims));
    tolerance::Float64=1e-8,
    maxiter::Int=200,
    sweepstrategy::Symbol=:backandforth,
    pivottolerance::Float64=1e-12,
    verbosity::Int=0,
    additionalpivots::Vector{MultiIndex}=MultiIndex[],
    normalizeerror::Bool=true
) where {ValueType,N}
    # 初始化TCI1
    tci = TensorCI1{ValueType}(f, localdims, firstpivot)
    n = length(tci)
    errors = Float64[]
    ranks = Int[]

    # 添加额外的初始枢轴
    for pivot in additionalpivots
        addglobalpivot!(tci, f, pivot, tolerance)
    end

    # 主迭代循环
    for iter in rank(tci)+1:maxiter
        # 根据扫描策略选择方向
        if forwardsweep(sweepstrategy, iter)
            # 正向扫描
            for bondindex in 1:n-1
                addpivot!(tci, bondindex, f, pivottolerance)
            end
        else
            # 反向扫描
            for bondindex in (n-1):-1:1
                addpivot!(tci, bondindex, f, pivottolerance)
            end
        end

        # 计算误差归一化因子
        errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
        push!(errors, lastsweeppivoterror(tci))
        push!(ranks, rank(tci))
        
        # 输出进度
        if verbosity > 0 && mod(iter, 10) == 0
            println("iteration = $iter, rank = $(last(ranks)), error= $(last(errors))")
        end
        
        # 检查收敛
        if last(errors) < tolerance * errornormalization
            break
        end
    end

    # 归一化误差并返回
    errornormalization = normalizeerror ? tci.maxsamplevalue : 1.0
    return tci, ranks, errors ./ errornormalization
end

@doc raw"""
    crossinterpolate(::Type{ValueType}, f, localdims, firstpivot; kwargs...) where {ValueType}

已废弃。请使用 [`crossinterpolate1`](@ref) 代替。

保留此函数仅为向后兼容。
"""
function crossinterpolate(
    ::Type{ValueType},
    f,
    localdims::Union{Vector{Int},NTuple{N,Int}},
    firstpivot::MultiIndex=ones(Int, length(localdims));
    kwargs...
) where {ValueType, N}
    return crossinterpolate1(ValueType, f, localdims, firstpivot; kwargs...)
end
