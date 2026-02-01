# ===================================================================
# integration.jl - 基于TCI的高维数值积分
# ===================================================================
# 这个文件实现了使用TCI和Gauss-Kronrod求积规则进行高维数值积分的功能。
#
# 高维积分是计算数学中的一个困难问题：
#   - 传统的网格方法受"维度诅咒"影响，计算量随维度指数增长
#   - TCI通过低秩近似可以有效处理某些类型的高维函数
#
# 工作原理：
#   1. 使用Gauss-Kronrod求积规则获取各维度的积分节点和权重
#   2. 定义一个带权重的函数，将积分转换为求和
#   3. 使用TCI构建这个带权函数的张量列表示
#   4. 对张量列求和得到积分结果
# ===================================================================

"""
    function integrate(
        ::Type{ValueType},
        f,
        a::Vector{ValueType},
        b::Vector{ValueType};
        tolerance=1e-8,
        GKorder::Int=15
    ) where {ValueType}

使用TCI和Gauss-Kronrod求积规则计算多维积分。

# 数学描述
计算以下积分：
```math
∫_{a_1}^{b_1} ∫_{a_2}^{b_2} ... ∫_{a_n}^{b_n} f(x_1, x_2, ..., x_n) dx_1 dx_2 ... dx_n
```

# 参数
- `ValueType`: 函数f的返回值类型（如Float64, ComplexF64等）
- `f`: 被积函数，接受一个坐标向量作为输入
- `a::Vector{ValueType}`: 各维度积分下界组成的向量
- `b::Vector{ValueType}`: 各维度积分上界组成的向量
- `GKorder::Int=15`: Gauss-Kronrod求积规则的阶数，必须是奇数
  - 常用值: 15 (默认), 21, 31, 41, 51, 61
  - 阶数越高精度越高，但计算量也越大
- `kwargs...`: 传递给crossinterpolate2的其他参数

# 返回值
- 积分的近似值

# 工作原理
1. 使用Gauss-Kronrod公式获取一维积分的节点和权重
2. 将积分区间[a_i, b_i]上的节点映射到[-1, 1]
3. 定义带权函数 F(indices) = w(indices) * f(x(indices))
4. 使用TCI构建F的张量列表示
5. 对张量列所有元素求和得到积分

# 示例
```julia
# 计算二维高斯函数的积分
f(x) = exp(-x[1]^2 - x[2]^2)
result = integrate(Float64, f, [-3.0, -3.0], [3.0, 3.0])
# 解析解约为 π ≈ 3.14159...
```

# 注意
- GKorder必须是奇数
- 积分上下界向量必须具有相同的长度
- tolerance参数控制TCI近似的精度，不是积分误差的直接控制
"""
function integrate(
    ::Type{ValueType},
    f,
    a::Vector{ValueType},
    b::Vector{ValueType};
    GKorder::Int=15,
    kwargs...  # 接受任意关键字参数并传递给crossinterpolate2
) where {ValueType}
    # ===== 参数验证 =====
    
    # Gauss-Kronrod阶数必须是奇数（这是算法的要求）
    if iseven(GKorder)
        error("Gauss--Kronrod order must be odd, e.g. 15 or 61.")
    end

    # 积分上下界的维度必须匹配
    if length(a) != length(b)
        error("Integral bounds must have the same dimensionality, but got $(length(a)) lower bounds and $(length(b)) upper bounds.")
    end

    # ===== 构建Gauss-Kronrod求积规则 =====
    
    # QuadGK.kronrod返回标准区间[-1, 1]上的节点和权重
    # 参数: GKorder ÷ 2 是Gauss点的数量（÷是整除运算符）
    # 返回值: nodes1d(节点), weights1d(权重), _ (误差估计，这里不使用)
    nodes1d, weights1d, _ = QuadGK.kronrod(GKorder ÷ 2, -1, +1)
    
    # ===== 将节点从[-1, 1]映射到[a, b] =====
    
    # 线性变换: x = (b-a) * (t+1)/2 + a，其中t ∈ [-1,1], x ∈ [a,b]
    # @. 宏表示对整个表达式进行广播(broadcast)
    # ' 是转置运算符，将nodes1d变成行向量
    # 结果nodes是一个 n_dims × n_nodes 的矩阵
    # nodes[i, j] 是第i维的第j个积分节点
    nodes = @. (b - a) * (nodes1d' + 1) / 2 + a
    
    # 权重也需要相应的缩放（雅可比行列式）
    # 当积分区间从[-1,1]变换到[a,b]时，权重需要乘以(b-a)/2
    weights = @. (b - a) * weights1d' / 2
    
    # 归一化因子：总节点数的幂次
    # 这是因为我们后面会对TCI结果求和，需要补偿
    normalization = GKorder^length(a)

    # ===== 定义局部维度 =====
    
    # localdims[i] = 第i维的节点数量
    # fill创建一个填充相同值的数组
    localdims = fill(length(nodes1d), length(a))

    # ===== 定义带权函数 =====
    
    """
    内部函数：将离散索引映射到实际坐标并计算带权函数值
    
    参数:
    - indices: 各维度的节点索引(1-based)
    
    返回值:
    - 带权重的函数值
    """
    function F(indices)
        # enumerate返回(位置, 值)对
        # nodes[n, i]是第n维的第i个节点坐标
        x = [nodes[n, i] for (n, i) in enumerate(indices)]
        
        # 计算组合权重（各维度权重的乘积）
        # prod计算迭代器中所有元素的乘积
        w = prod(weights[n, i] for (n, i) in enumerate(indices))
        
        # 返回 权重 × 函数值 × 归一化因子
        # 归一化因子确保sum(tci)能直接给出积分结果
        return w * f(x) * normalization
    end

    # ===== 使用TCI2构建张量列 =====
    
    # crossinterpolate2是TCI2算法的主入口
    # nsearchglobalpivot=10 设置全局枢轴搜索的次数
    # kwargs...传递其他参数如tolerance
    tci2, ranks, errors = crossinterpolate2(
        ValueType,
        F,
        localdims;
        nsearchglobalpivot=10,
        kwargs...
    )

    # ===== 计算积分结果 =====
    
    # sum(tci2)计算张量列表示的函数在所有格点上的和
    # 除以normalization得到真正的积分值
    return sum(tci2) / normalization
end
