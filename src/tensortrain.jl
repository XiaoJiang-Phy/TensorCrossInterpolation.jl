# ===================================================================
# tensortrain.jl - 张量列类型及其操作
# ===================================================================
# 这个文件定义了TensorTrain类型，它是张量列的具体实现。
#
# 张量列(Tensor Train, TT)，也称为矩阵乘积态(Matrix Product State, MPS)，
# 是高维张量的低秩表示：
#   f(i₁, i₂, ..., iₙ) = T₁[i₁] * T₂[i₂] * ... * Tₙ[iₙ]
# 其中每个 Tₖ[iₖ] 是一个矩阵（第一个和最后一个是向量）。
#
# 这种表示将指数级别的存储 O(d^n) 降低到多项式级别 O(n * d * r²)
# 其中 d 是局部维度，r 是键维度（bond dimension）。
# ===================================================================

"""
    struct TensorTrain{ValueType, N} <: AbstractTensorTrain{ValueType}

张量列的数据结构。

# 类型参数
- `ValueType`: 张量元素的类型（如 Float64, ComplexF64）
- `N`: 每个张量核心的维度数
  - `N=3` 表示标准的张量列：(左键, 站点索引, 右键)
  - `N>3` 表示每个站点有多个索引

# 字段
- `sitetensors::Vector{Array{ValueType,N}}`: 张量核心列表

# 张量结构
每个张量 Tₖ 的形状是 (χₖ₋₁, d₁, d₂, ..., dₘ, χₖ)
其中：
- χₖ₋₁, χₖ 是键维度（边界：χ₀ = χₙ = 1）
- d₁, ..., dₘ 是站点维度（物理指标）

# 求值
张量列可以像函数一样调用：
```julia
tt = TensorTrain(...)
value = tt([1, 2, 3, 4])  # 计算 f(1, 2, 3, 4)
```

# 示例
```julia
# 创建随机张量列
tensors = [
    rand(1, 3, 5),    # 第1个张量：(1, 3, 5)
    rand(5, 3, 8),    # 第2个张量：(5, 3, 8)
    rand(8, 3, 4),    # 第3个张量：(8, 3, 4)
    rand(4, 3, 1)     # 第4个张量：(4, 3, 1)
]
tt = TensorTrain(tensors)
println(tt([1, 2, 3, 2]))  # 求值
```

# 另见
参见: [`AbstractTensorTrain`](@ref), [`TensorCI1`](@ref), [`TensorCI2`](@ref)
"""
mutable struct TensorTrain{ValueType,N} <: AbstractTensorTrain{ValueType}
    sitetensors::Vector{Array{ValueType,N}}

    """
        TensorTrain{ValueType,N}(sitetensors) where {ValueType,N}
    
    内部构造函数，验证张量维度兼容性。
    """
    function TensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}) where {ValueType,N}
        # 验证相邻张量的键维度匹配
        for i in 1:length(sitetensors)-1
            # 第i个张量的最后一个维度应等于第i+1个张量的第一个维度
            if (last(size(sitetensors[i])) != size(sitetensors[i+1], 1))
                throw(ArgumentError(
                    "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
                ))
            end
        end

        new{ValueType,N}(sitetensors)
    end
end

"""
    Base.show(io::IO, obj::TensorTrain{V,N}) where {V,N}

定义TensorTrain的打印格式。
"""
function Base.show(io::IO, obj::TensorTrain{V,N}) where {V,N}
    print(
        io,
        "$(typeof(obj)) of rank $(maximum(linkdims(obj)))"
    )
end

"""
    TensorTrain{V2,N}(tt::TensorTrain{V}) where {V,V2,N}

将TensorTrain转换为不同类型的TensorTrain。

# 用途
类型转换，例如将Float64转换为ComplexF64。
"""
function TensorTrain{V2,N}(tt::TensorTrain{V})::TensorTrain{V2,N} where {V,V2,N}
    # Array{V2}.(arr) 将每个张量转换为类型V2
    return TensorTrain{V2,N}(Array{V2}.(sitetensors(tt)))
end

"""
    TensorTrain(sitetensors::AbstractVector{<:AbstractArray{V,N}}) where {V,N}

从张量列表创建TensorTrain（简化构造函数）。

# 参数
- `sitetensors`: 张量核心的数组

# 说明
每个张量应该有以下结构：
- 第1维：左键（连接到前一个张量）
- 中间维度：站点索引（"物理指标"）
- 最后一维：右键（连接到下一个张量）

# 示例
```julia
tensors = [rand(1, d, 5) for d in [3, 4, 5]]
tensors[end] = rand(5, 6, 1)  # 最后一个张量的右键维度为1
tt = TensorTrain(tensors)
```
"""
function TensorTrain(sitetensors::AbstractVector{<:AbstractArray{V,N}}) where {V,N}
    return TensorTrain{V,N}(sitetensors)
end

"""
    TensorTrain(tci::AbstractTensorTrain{V}) where {V}

将任何AbstractTensorTrain对象转换为TensorTrain。

# 用途
将TCI对象（TensorCI1, TensorCI2）转换为标准TensorTrain。

# 示例
```julia
tci, _, _ = crossinterpolate2(Float64, f, [10, 10, 10])
tt = TensorTrain(tci)  # 转换为TensorTrain
```
"""
function TensorTrain(tci::AbstractTensorTrain{V})::TensorTrain{V,3} where {V}
    return TensorTrain{V,3}(sitetensors(tci))
end

"""
    TensorTrain{V2,N}(tt::AbstractTensorTrain{V}, localdims) where {V,V2,N}

创建具有指定站点维度的TensorTrain。

# 参数
- `tt`: 源张量列
- `localdims`: 每个站点的维度列表，每个元素是一个含有N-2个整数的数组

# 用途
重塑张量，使每个站点有多个索引。
"""
function TensorTrain{V2,N}(tt::AbstractTensorTrain{V}, localdims)::TensorTrain{V2,N} where {V,V2,N}
    for d in localdims
        length(d) == N - 2 || error("Each element of localdims be a list of N-2 integers.")
    end
    for n in 1:length(tt)
        prod(size(tt[n])[2:end-1]) == prod(localdims[n]) || error("The local dimensions at n=$n must match the tensor sizes.")
    end
    return TensorTrain{V2,N}(
        [reshape(Array{V2}(t), size(t, 1), localdims[n]..., size(t)[end]) for (n, t) in enumerate(sitetensors(tt))])
end

function TensorTrain{N}(tt::AbstractTensorTrain{V}, localdims)::TensorTrain{V,N} where {V,N}
    return TensorTrain{V,N}(tt, localdims)
end

"""
    tensortrain(tci)

将TCI对象转换为TensorTrain的便捷函数。

# 示例
```julia
tci, _, _ = crossinterpolate2(Float64, f, [10, 10, 10])
tt = tensortrain(tci)
```
"""
function tensortrain(tci)
    return TensorTrain(tci)
end

# ===================================================================
# 压缩相关函数
# ===================================================================

"""
    _factorize(A::AbstractMatrix{V}, method::Symbol; tolerance, maxbonddim, leftorthogonal, normalizeerror) where {V}

将矩阵分解为两个因子（内部函数）。

# 参数
- `A`: 要分解的矩阵
- `method`: 分解方法 (:LU, :CI, :SVD)
- `tolerance`: 截断容差
- `maxbonddim`: 最大秩/键维度
- `leftorthogonal`: 是否左正交化
- `normalizeerror`: 是否归一化误差

# 返回值
- `(left, right, rank)` 元组
  - `left`: 左因子矩阵
  - `right`: 右因子矩阵
  - `rank`: 因子的秩

# 分解方法
- `:LU`: 秩揭示LU分解，速度快但精度略低
- `:CI`: 交叉插值分解
- `:SVD`: 奇异值分解，精度最高但计算量大
"""
function _factorize(
    A::AbstractMatrix{V}, method::Symbol; tolerance::Float64, maxbonddim::Int, leftorthogonal::Bool=false, normalizeerror=true
)::Tuple{Matrix{V},Matrix{V},Int} where {V}
    # 设置容差
    reltol = 1e-14
    abstol = 0.0
    if normalizeerror
        reltol = tolerance
    else
        abstol = tolerance
    end
    
    # 根据方法选择不同的分解
    if method === :LU
        # 使用秩揭示LU分解
        factorization = rrlu(A, abstol=abstol, reltol=reltol, maxrank=maxbonddim, leftorthogonal=leftorthogonal)
        return left(factorization), right(factorization), npivots(factorization)
    elseif method === :CI
        # 使用交叉插值
        factorization = MatrixLUCI(A, abstol=abstol, reltol=reltol, maxrank=maxbonddim, leftorthogonal=leftorthogonal)
        return left(factorization), right(factorization), npivots(factorization)
    elseif method === :SVD
        # 使用奇异值分解
        factorization = LinearAlgebra.svd(A)
        
        # 计算截断误差：保留前n个奇异值后的误差
        err = [sum(factorization.S[n+1:end] .^ 2) for n in 1:length(factorization.S)]
        normalized_err = err ./ sum(factorization.S .^ 2)

        # 确定截断位置
        trunci = min(
            replacenothing(findfirst(<(abstol^2), err), length(err)),
            replacenothing(findfirst(<(reltol^2), normalized_err), length(normalized_err)),
            maxbonddim
        )
        
        # 构造左右因子
        if leftorthogonal
            return (
                factorization.U[:, 1:trunci],  # U是正交的
                Diagonal(factorization.S[1:trunci]) * factorization.Vt[1:trunci, :],
                trunci
            )
        else
            return (
                factorization.U[:, 1:trunci] * Diagonal(factorization.S[1:trunci]),
                factorization.Vt[1:trunci, :],  # V是正交的
                trunci
            )
        end
    else
        error("Not implemented yet.")
    end
end

"""
    compress!(tt::TensorTrain{V,N}, method::Symbol=:LU; tolerance=1e-12, maxbonddim=typemax(Int)) where {V,N}

原地压缩张量列。

# 参数
- `tt`: 要压缩的张量列（会被修改）
- `method`: 分解方法 (:LU, :CI, 或 :SVD)
- `tolerance`: 压缩容差
- `maxbonddim`: 最大键维度

# 算法（两遍扫描）
1. 从左到右扫描：将每个张量分解，左因子留在当前位置，右因子吸收到下一个张量
2. 从右到左扫描：将每个张量分解并截断，右因子留在当前位置，左因子吸收到前一个张量

# 结果
压缩后的张量列满足：
- 键维度不超过maxbonddim
- 截断误差满足tolerance要求
- 张量列处于右正交规范形式

# 示例
```julia
tt = TensorTrain(...)
compress!(tt, :SVD; tolerance=1e-8, maxbonddim=50)
println("压缩后的秩: $(rank(tt))")
```
"""
function compress!(
    tt::TensorTrain{V,N},
    method::Symbol=:LU;
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
    normalizeerror::Bool=true
) where {V,N}
    # ===== 第一遍：从左到右（不截断，只正交化）=====
    for ell in 1:length(tt)-1
        shapel = size(tt.sitetensors[ell])
        
        # 将张量重塑为矩阵：(左键*站点维度) × 右键
        left, right, newbonddim = _factorize(
            reshape(tt.sitetensors[ell], prod(shapel[1:end-1]), shapel[end]),
            method; tolerance=0.0, maxbonddim=typemax(Int), leftorthogonal=true  # 无截断
        )
        
        # 左因子放回当前位置
        tt.sitetensors[ell] = reshape(left, shapel[1:end-1]..., newbonddim)
        
        # 右因子吸收到下一个张量
        shaper = size(tt.sitetensors[ell+1])
        nexttensor = right * reshape(tt.sitetensors[ell+1], shaper[1], prod(shaper[2:end]))
        tt.sitetensors[ell+1] = reshape(nexttensor, newbonddim, shaper[2:end]...)
    end

    # ===== 第二遍：从右到左（截断）=====
    for ell in length(tt):-1:2
        shaper = size(tt.sitetensors[ell])
        
        # 将张量重塑为矩阵：左键 × (站点维度*右键)
        left, right, newbonddim = _factorize(
            reshape(tt.sitetensors[ell], shaper[1], prod(shaper[2:end])),
            method; tolerance, maxbonddim, normalizeerror, leftorthogonal=false  # 执行截断
        )
        
        # 右因子放回当前位置
        tt.sitetensors[ell] = reshape(right, newbonddim, shaper[2:end]...)
        
        # 左因子吸收到前一个张量
        shapel = size(tt.sitetensors[ell-1])
        nexttensor = reshape(tt.sitetensors[ell-1], prod(shapel[1:end-1]), shapel[end]) * left
        tt.sitetensors[ell-1] = reshape(nexttensor, shapel[1:end-1]..., newbonddim)
    end

    nothing  # 返回nothing表示这是原地修改
end

# ===================================================================
# 标量乘法和除法
# ===================================================================

"""
    multiply!(tt::TensorTrain{V,N}, a) where {V,N}

将张量列原地乘以标量。
"""
function multiply!(tt::TensorTrain{V,N}, a) where {V,N}
    # .= 是广播赋值，.* 是广播乘法
    tt.sitetensors[end] .= tt.sitetensors[end] .* a
    nothing
end

function multiply!(a, tt::TensorTrain{V,N}) where {V,N}
    tt.sitetensors[end] .= a .* tt.sitetensors[end]
    nothing
end

"""
    multiply(tt::TensorTrain{V,N}, a)::TensorTrain{V,N} where {V,N}

返回张量列与标量的乘积（不修改原对象）。
"""
function multiply(tt::TensorTrain{V,N}, a)::TensorTrain{V,N} where {V,N}
    tt2 = deepcopy(tt)
    multiply!(tt2, a)
    return tt2
end

function multiply(a, tt::TensorTrain{V,N})::TensorTrain{V,N} where {V,N}
    tt2 = deepcopy(tt)
    multiply!(a, tt2)
    return tt2
end

"""
    Base.:*(tt::TensorTrain{V,N}, a) where {V,N}
    Base.:*(a, tt::TensorTrain{V,N}) where {V,N}

张量列与标量的乘法运算符。

# 示例
```julia
tt2 = tt * 2.0
tt3 = 3.0 * tt
```
"""
function Base.:*(tt::TensorTrain{V,N}, a)::TensorTrain{V,N} where {V,N}
    return multiply(tt, a)
end

function Base.:*(a, tt::TensorTrain{V,N})::TensorTrain{V,N} where {V,N}
    return multiply(a, tt)
end

"""
    divide!(tt::TensorTrain{V,N}, a) where {V,N}

将张量列原地除以标量。
"""
function divide!(tt::TensorTrain{V,N}, a) where {V,N}
    tt.sitetensors[end] .= tt.sitetensors[end] ./ a
    nothing
end

"""
    divide(tt::TensorTrain{V,N}, a) where {V,N}

返回张量列除以标量的结果（不修改原对象）。
"""
function divide(tt::TensorTrain{V,N}, a) where {V,N}
    tt2 = deepcopy(tt)
    divide!(tt2, a)
    return tt2
end

"""
    Base.:/(tt::TensorTrain{V,N}, a) where {V,N}

张量列除法运算符。
"""
function Base.:/(tt::TensorTrain{V,N}, a) where {V,N}
    return divide(tt, a)
end

"""
    Base.reverse(tt::AbstractTensorTrain{V}) where {V}

反转张量列的顺序。

# 说明
返回一个新的张量列，其中张量的顺序被反转。
每个张量的第一维和最后一维交换。

# 示例
```julia
# 如果 tt 表示 f(i₁, i₂, i₃)
# 那么 reverse(tt) 表示 f(i₃, i₂, i₁)
```
"""
function Base.reverse(tt::AbstractTensorTrain{V}) where {V}
    return tensortrain(reverse([
        # permutedims 交换第1维和最后一维
        permutedims(T, (ndims(T), (2:ndims(T)-1)..., 1)) for T in sitetensors(tt)
    ]))
end

# ===================================================================
# TensorTrainFit - 张量列拟合
# ===================================================================

"""
    TensorTrainFit{ValueType}

用于拟合数据的张量列包装器。

# 背景
当被插值的数据有噪声时，直接插值可能不是最佳选择。
TensorTrainFit允许使用优化方法（如最小二乘）来拟合张量列。

# 字段
- `indexsets::Vector{MultiIndex}`: 数据点的索引
- `values::Vector{ValueType}`: 数据点的值
- `tt::TensorTrain{ValueType,3}`: 初始张量列
- `offsets::Vector{Int}`: 用于参数展平的偏移量

# 用途
1. 创建初始张量列
2. 使用优化器最小化拟合误差
3. 从优化参数重建张量列
"""
struct TensorTrainFit{ValueType}
    indexsets::Vector{MultiIndex}
    values::Vector{ValueType}
    tt::TensorTrain{ValueType,3}
    offsets::Vector{Int}
end

"""
    TensorTrainFit{ValueType}(indexsets, values, tt) where {ValueType}

创建TensorTrainFit对象。

# 参数
- `indexsets`: 数据点索引的列表
- `values`: 对应的值
- `tt`: 初始张量列猜测
"""
function TensorTrainFit{ValueType}(indexsets, values, tt) where {ValueType}
    # 计算每个张量的参数偏移量
    offsets = [0]
    for n in 1:length(tt)
        push!(offsets, offsets[end] + length(tt[n]))
    end
    return TensorTrainFit{ValueType}(indexsets, values, tt, offsets)
end

"""
    flatten(obj::TensorTrain{ValueType}) where {ValueType}

将张量列的所有参数展平为一维向量。

# 用途
用于优化：将张量列参数化为单个向量。
"""
flatten(obj::TensorTrain{ValueType}) where {ValueType} = vcat(vec.(sitetensors(obj))...)

"""
    to_tensors(obj::TensorTrainFit{ValueType}, x::Vector{ValueType}) where {ValueType}

从展平的参数向量重建张量列表。
"""
function to_tensors(obj::TensorTrainFit{ValueType}, x::Vector{ValueType}) where {ValueType}
    return [
        reshape(
            x[obj.offsets[n]+1:obj.offsets[n+1]],
            size(obj.tt[n])
        )
        for n in 1:length(obj.tt)
    ]
end

"""
    _evaluate(tt::Vector{Array{V,3}}, indexset) where {V}

辅助函数：在给定张量列表上求值。
"""
function _evaluate(tt::Vector{Array{V,3}}, indexset) where {V}
    only(prod(T[:, i, :] for (T, i) in zip(tt, indexset)))
end

"""
    (obj::TensorTrainFit{ValueType})(x::Vector{ValueType}) where {ValueType}

计算拟合误差（残差平方和）。

# 用途
作为优化的目标函数。

# 返回值
∑ᵢ |tt(indexsets[i]) - values[i]|²
"""
function (obj::TensorTrainFit{ValueType})(x::Vector{ValueType}) where {ValueType}
    tensors = to_tensors(obj, x)
    return sum((abs2(_evaluate(tensors, indexset) - obj.values[i]) for (i, indexset) in enumerate(obj.indexsets)))
end

# ===================================================================
# 完整张量转换
# ===================================================================

"""
    fulltensor(obj::TensorTrain{T,N})::Array{T} where {T,N}

将张量列转换为完整的多维数组。

# 警告
完整张量的大小是指数级的！只有在维度数量很小时才使用。

# 返回值
- 完整的多维数组，形状为 (d₁, d₂, ..., dₙ)

# 示例
```julia
tt = TensorTrain(...)  # 假设表示一个3×4×5的张量
A = fulltensor(tt)     # 返回 3×4×5 的Array
```
"""
function fulltensor(obj::TensorTrain{T,N})::Array{T} where {T,N}
    sitedims_ = sitedims(obj)
    localdims = collect(prod.(sitedims_))
    
    # 从第一个张量开始，逐个收缩
    result::Matrix{T} = reshape(obj.sitetensors[1], localdims[1], :)
    leftdim = localdims[1]
    
    for l in 2:length(obj)
        # 重塑下一个张量
        nextmatrix = reshape(
            obj.sitetensors[l], size(obj.sitetensors[l], 1), localdims[l] * size(obj.sitetensors[l])[end])
        leftdim *= localdims[l]
        # 收缩并重塑
        result = reshape(result * nextmatrix, leftdim, size(obj.sitetensors[l])[end])
    end
    
    # 最终重塑
    returnsize = collect(Iterators.flatten(sitedims_))
    return reshape(result, returnsize...)
end
