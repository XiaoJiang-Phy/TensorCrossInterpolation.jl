# ===================================================================
# conversion.jl - 类型转换函数
# ===================================================================
# 这个文件提供不同TCI表示之间的转换函数。
#
# TensorCrossInterpolation包有多种表示张量列的方式：
#   - TensorCI1: TCI1算法产生的表示
#   - TensorCI2: TCI2算法产生的表示
#   - TensorTrain: 标准张量列
#   - MatrixACA: 自适应交叉近似
#   - rrLU: 秩揭示LU分解
#
# 这些转换函数允许在不同表示之间灵活转换，
# 以便利用各种算法和功能。
# ===================================================================

"""
    MatrixACA(lu::rrLU{T}) where {T}

将秩揭示LU分解转换为MatrixACA格式。

# 参数
- `lu::rrLU{T}`: 秩揭示LU分解对象

# 返回值
- `MatrixACA{T}` 对象

# 数学说明
LU分解 A = L * U 可以表示为 ACA 格式：
A ≈ ∑ₖ αₖ uₖ vₖᵀ

其中 uₖ 来自 L，vₖ 来自 U，αₖ = 1/对角元素。

# 左正交和右正交的区别
- 左正交时：u 矩阵包含对角因子
- 非左正交时：v 矩阵包含对角因子

# 示例
```julia
A = rand(100, 50)
lu = rrlu(A)
aca = MatrixACA(lu)  # 转换为ACA格式
```
"""
function MatrixACA(lu::rrLU{T}) where {T}
    # 创建空的ACA对象
    aca = MatrixACA(T, size(lu)...)
    
    # 复制枢轴索引
    aca.rowindices = rowindices(lu)
    aca.colindices = colindices(lu)
    
    # 复制左右因子
    aca.u = left(lu)   # L 矩阵部分
    aca.v = right(lu)  # U 矩阵部分
    
    # 计算 alpha 权重（对角线的倒数）
    aca.alpha = 1 ./ diag(lu)

    # 根据正交化方向调整矩阵
    if lu.leftorthogonal
        # 左正交时：将对角因子乘入 u 矩阵
        for j in axes(aca.u, 2)
            aca.u[:, j] *= diag(lu)[j]
        end
    else
        # 非左正交时：将对角因子乘入 v 矩阵
        for i in axes(aca.v, 1)
            aca.v[i, :] *= diag(lu)[i]
        end
    end
    
    return aca
end

"""
    TensorCI1{ValueType}(tci2::TensorCI2{ValueType}, f; kwargs...) where {ValueType}

将TensorCI2对象转换为TensorCI1格式。

# 参数
- `tci2::TensorCI2{ValueType}`: 源TCI2对象
- `f`: 原始函数（用于重建Pi矩阵）
- `kwargs...`: 其他参数

# 返回值
- `TensorCI1{ValueType}` 对象

# 说明
TCI1和TCI2有不同的内部表示：
- TCI1 存储 T 和 P 张量，以及完整的 Pi 矩阵
- TCI2 存储站点张量和索引集

转换需要重新计算 Pi 矩阵，因此需要原始函数 f。

# 用途
当需要使用 TCI1 特有的功能（如增量更新）时使用。
"""
function TensorCI1{ValueType}(
    tci2::TensorCI2{ValueType},
    f;
    kwargs...
) where {ValueType}
    L = length(tci2)
    
    # 创建空的TCI1对象
    tci1 = TensorCI1{ValueType}(tci2.localdims)
    
    # 将TCI2的索引集转换为IndexSet格式
    tci1.Iset = IndexSet.(tci2.Iset)
    tci1.Jset = IndexSet.(tci2.Jset)
    
    # 重新计算 Pi 索引集
    # getPiIset.(Ref(tci1), 1:L) 对范围内的每个值调用 getPiIset
    # Ref(tci1) 确保 tci1 不被广播
    tci1.PiIset = getPiIset.(Ref(tci1), 1:L)
    tci1.PiJset = getPiJset.(Ref(tci1), 1:L)
    
    # 重新计算 Pi 矩阵（需要调用原始函数）
    tci1.Pi = getPi.(Ref(tci1), 1:L-1, f)

    # 从 Pi 矩阵重建 T 和 P 张量
    for ell in 1:L-1
        # 获取沿 Pi 轴的索引以重建 T, P
        iset = pos(tci1.PiIset[ell], tci1.Iset[ell+1].fromint)
        jset = pos(tci1.PiJset[ell+1], tci1.Jset[ell].fromint)
        
        # 更新 T 张量
        updateT!(tci1, ell, tci1.Pi[ell][:, jset])
        if ell == L - 1
            # 最后一个张量特殊处理
            updateT!(tci1, L, tci1.Pi[ell][iset, :])
        end
        
        # 设置 P 矩阵（枢轴矩阵）
        tci1.P[ell] = tci1.Pi[ell][iset, jset]

        # 创建对应的 ACA 对象
        tci1.aca[ell] = MatrixACA(tci1.Pi[ell], (iset[1], jset[1]))
        # 添加其余枢轴
        for (rowindex, colindex) in zip(iset[2:end], jset[2:end])
            addpivotcol!(tci1.aca[ell], tci1.Pi[ell], colindex)
            addpivotrow!(tci1.aca[ell], tci1.Pi[ell], rowindex)
        end
    end
    
    # 最后一个 P 矩阵是 1×1 的单位矩阵
    tci1.P[end] = ones(ValueType, 1, 1)

    # 复制误差信息
    tci1.pivoterrors = tci2.bonderrors
    tci1.maxsamplevalue = tci2.maxsamplevalue
    
    return tci1
end

"""
    TensorCI2{ValueType}(tci1::TensorCI1{ValueType}) where {ValueType}

将TensorCI1对象转换为TensorCI2格式。

# 参数
- `tci1::TensorCI1{ValueType}`: 源TCI1对象

# 返回值
- `TensorCI2{ValueType}` 对象

# 说明
这个转换不需要原始函数，因为所有需要的信息都在 TCI1 对象中。

# 转换过程
1. 创建新的 TCI2 对象
2. 将 IndexSet 格式的索引集转换为普通向量
3. 计算站点张量：Tₖ * P⁻¹ₖ
4. 复制误差信息
"""
function TensorCI2{ValueType}(tci1::TensorCI1{ValueType}) where {ValueType}
    # 创建TCI2对象
    # vcat(collect.(sitedims(tci1))...) 将站点维度展平并连接
    tci2 = TensorCI2{ValueType}(vcat(collect.(sitedims(tci1))...)::Vector{Int})
    
    # 将 IndexSet 转换为普通向量
    tci2.Iset = [i.fromint for i in tci1.Iset]
    tci2.Jset = [j.fromint for j in tci1.Jset]
    tci2.localdims = tci1.localdims
    
    L = length(tci1)
    
    # 计算站点张量
    # TtimesPinv 计算 T * P⁻¹
    tci2.sitetensors[1:L-1] = TtimesPinv.(tci1, 1:L-1)
    tci2.sitetensors[end] = tci1.T[end]
    
    # 复制误差信息
    tci2.pivoterrors = Float64[]
    tci2.bonderrors = tci1.pivoterrors
    tci2.maxsamplevalue = tci1.maxsamplevalue
    
    return tci2
end

"""
    sweep1sitegetindices!(tt::TensorTrain{ValueType,3}, forwardsweep::Bool, spectatorindices; maxbonddim, tolerance) where {ValueType}

对张量列进行单站点扫描并提取索引集（内部函数）。

# 参数
- `tt`: 张量列（会被修改为正交形式）
- `forwardsweep`: true 为正向扫描，false 为反向扫描
- `spectatorindices`: 可选的旁观者索引
- `maxbonddim`: 最大键维度
- `tolerance`: 截断容差

# 返回值
- `(indexset, pivoterrors)` 元组

# 算法
对每个键进行 LU-CI 分解，提取枢轴索引并更新张量。
这是将 TensorTrain 转换为 TCI2 的核心步骤。
"""
function sweep1sitegetindices!(
    tt::TensorTrain{ValueType,3}, forwardsweep::Bool,
    spectatorindices::Vector{Vector{MultiIndex}}=Vector{MultiIndex}[];
    maxbonddim=typemax(Int), tolerance=0.0
) where {ValueType}
    # 初始化索引集（从空索引开始）
    indexset = Vector{MultiIndex}[MultiIndex[[]]]
    pivoterrorsarray = zeros(rank(tt) + 1)

    # 内部函数：根据扫描方向重塑张量为矩阵
    function groupindices(T::AbstractArray, next::Bool)
        shape = size(T)
        if forwardsweep != next
            # 将站点维度与左键维度合并
            reshape(T, prod(shape[1:end-1]), shape[end])
        else
            # 将站点维度与右键维度合并
            reshape(T, shape[1], prod(shape[2:end]))
        end
    end

    # 内部函数：将矩阵重塑回张量
    function splitindices(T::AbstractArray, shape, newbonddim, next::Bool)
        if forwardsweep != next
            newshape = (shape[1:end-1]..., newbonddim)
        else
            newshape = (newbonddim, shape[2:end]...)
        end
        reshape(T, newshape)
    end

    L = length(tt)
    
    # 扫描张量列
    for i in 1:L-1
        # 根据扫描方向确定当前和下一个位置
        ell = forwardsweep ? i : L - i + 1
        ellnext = forwardsweep ? i + 1 : L - i
        shape = size(tt.sitetensors[ell])
        shapenext = size(tt.sitetensors[ellnext])

        # 对当前张量进行 LU-CI 分解
        luci = MatrixLUCI(
            groupindices(tt.sitetensors[ell], false), leftorthogonal=forwardsweep,
            abstol=tolerance, maxrank=maxbonddim
        )

        # 更新索引集
        if forwardsweep
            # 正向扫描：更新 I 集
            push!(indexset, kronecker(last(indexset), shape[2])[rowindices(luci)])
            if !isempty(spectatorindices)
                spectatorindices[ell] = spectatorindices[ell][colindices(luci)]
            end
        else
            # 反向扫描：更新 J 集
            push!(indexset, kronecker(shape[2], last(indexset))[colindices(luci)])
            if !isempty(spectatorindices)
                spectatorindices[ell] = spectatorindices[ell][rowindices(luci)]
            end
        end

        # 更新当前张量
        tt.sitetensors[ell] = splitindices(
            forwardsweep ? left(luci) : right(luci),
            shape, npivots(luci), false
        )

        # 将另一个因子吸收到下一个张量
        nexttensor = (
            forwardsweep
            ? right(luci) * groupindices(tt.sitetensors[ellnext], true)
            : groupindices(tt.sitetensors[ellnext], true) * left(luci)
        )

        tt.sitetensors[ellnext] = splitindices(nexttensor, shapenext, npivots(luci), true)
        
        # 更新误差数组
        pivoterrorsarray[1:npivots(luci) + 1] = max.(pivoterrorsarray[1:npivots(luci) + 1], pivoterrors(luci))
    end

    # 根据扫描方向可能需要反转索引集
    if forwardsweep
        return indexset, pivoterrorsarray
    else
        return reverse(indexset), pivoterrorsarray
    end
end

"""
    TensorCI2{ValueType}(tt::TensorTrain{ValueType,3}; tolerance=1e-12, maxbonddim=typemax(Int), maxiter=3) where {ValueType}

将TensorTrain对象转换为TensorCI2格式。

# 参数
- `tt::TensorTrain{ValueType,3}`: 源张量列
- `tolerance`: 截断容差
- `maxbonddim`: 最大键维度
- `maxiter`: 最大迭代次数

# 返回值
- `TensorCI2{ValueType}` 对象

# 算法
1. 正向扫描提取 I 索引集
2. 反向扫描提取 J 索引集
3. 交替扫描直到收敛或达到最大迭代次数

# 转换过程
将普通张量列转换为 TCI2 格式，需要确定合适的枢轴索引集。
这通过交替的正向和反向扫描来完成。

# 示例
```julia
tt = TensorTrain(...)
tci2 = TensorCI2{Float64}(tt; tolerance=1e-10)
# 现在可以使用 TCI2 的功能，如添加全局枢轴等
```
"""
function TensorCI2{ValueType}(
    tt::TensorTrain{ValueType,3}; tolerance=1e-12, maxbonddim=typemax(Int), maxiter=3
) where {ValueType}
    local pivoterrors::Vector{Float64}

    # 第一遍：正向扫描获取 I 集
    Iset, = sweep1sitegetindices!(tt, true; maxbonddim, tolerance)
    
    # 第二遍：反向扫描获取 J 集
    Jset, pivoterrors = sweep1sitegetindices!(tt, false; maxbonddim, tolerance)

    # 交替扫描直到收敛
    for iter in 3:maxiter
        if isodd(iter)
            # 奇数次迭代：正向扫描更新 I 集
            Isetnew, pivoterrors = sweep1sitegetindices!(tt, true, Jset)
            if Isetnew == Iset
                break  # 收敛
            end
        else
            # 偶数次迭代：反向扫描更新 J 集
            Jsetnew, pivoterrors = sweep1sitegetindices!(tt, false, Iset)
            if Jsetnew == Jset
                break  # 收敛
            end
        end
    end

    # 创建 TCI2 对象
    tci2 = TensorCI2{ValueType}(first.(sitedims(tt)))
    tci2.Iset = Iset
    tci2.Jset = Jset
    tci2.sitetensors = sitetensors(tt)
    tci2.pivoterrors = pivoterrors
    
    # 估计最大采样值
    tci2.maxsamplevalue = maximum(maximum.(abs, tci2.sitetensors))

    return tci2
end
