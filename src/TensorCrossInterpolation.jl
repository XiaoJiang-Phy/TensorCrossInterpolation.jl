# ===================================================================
# TensorCrossInterpolation.jl - 主模块文件
# ===================================================================
# 这是TensorCrossInterpolation包的入口点和主模块定义文件。
# 
# 张量交叉插值(Tensor Cross Interpolation, TCI)是一种用于高效表示和
# 近似高维函数的方法。它将复杂的多变量函数分解为低秩张量列
# (Tensor Train, TT)格式，从而大幅减少存储和计算成本。
#
# 本包提供了两种TCI算法的实现:
#   - TCI1: 原始的TCI算法，适用于大多数应用场景
#   - TCI2: 改进版TCI算法，具有更好的收敛性和稳定性
#
# 主要功能:
#   - 将任意多变量函数插值为张量列格式
#   - 张量列的算术运算（加法、乘法、收缩等）
#   - 张量列的压缩和优化
#   - 高维积分的数值计算
# ===================================================================

# 定义模块，module关键字声明一个新的命名空间
module TensorCrossInterpolation

# ===================================================================
# 依赖包导入
# ===================================================================

# LinearAlgebra: Julia标准库，提供线性代数功能
# 包括矩阵分解(LU, SVD, QR等)、范数计算、矩阵运算等
using LinearAlgebra

# EllipsisNotation: 提供类似NumPy的省略号(...)语法
# 用于多维数组的切片操作，例如 A[.., 1] 表示 A[:,:,...,:,1]
using EllipsisNotation

# BitIntegers: 提供大整数类型支持
# 当函数参数空间很大时，需要使用UInt256等大整数来编码索引
using BitIntegers

# QuadGK: 自适应Gauss-Kronrod数值积分
# 用于高精度一维数值积分，在多维积分中用于各个维度
import QuadGK

# ===================================================================
# 从其他模块导入特定函数并扩展
# ===================================================================

# 从LinearAlgebra导入rank和diag函数
# 使用import而非using是因为我们要为自定义类型添加新方法(method)
# rank: 计算矩阵/张量的秩
# diag: 提取对角线元素
import LinearAlgebra: rank, diag

# 为LinearAlgebra创建别名LA，方便后续使用
import LinearAlgebra as LA

# 从Base模块导入相等性比较运算符和加法运算符
# 我们需要为自定义类型(如IndexSet、TensorTrain)定义这些运算
import Base: ==, +

# 从Base导入迭代器相关函数，用于定义自定义类型的迭代行为
# isempty: 判断容器是否为空
# iterate: 定义迭代协议（for循环的底层机制）
# getindex: 定义[]索引访问（如 obj[1]）
# lastindex: 定义end关键字的值（如 obj[end]）
# broadcastable: 定义广播行为（如 f.(obj)）
import Base: isempty, iterate, getindex, lastindex, broadcastable

# 从Base导入尺寸和聚合相关函数
# length: 获取容器长度
# size: 获取数组维度
# sum: 求和运算
import Base: length, size, sum

# 导入Random模块用于随机数生成
import Random

# ===================================================================
# 公开API (export)
# ===================================================================
# 这些是包的公开接口，用户可以直接使用而无需加模块前缀

# 主要插值函数
# crossinterpolate1: TCI1算法的主入口
# crossinterpolate2: TCI2算法的主入口  
# optfirstpivot: 优化初始枢轴点的辅助函数
export crossinterpolate1, crossinterpolate2, optfirstpivot

# 张量列相关类型和函数
# tensortrain: 将TCI对象转换为TensorTrain
# TensorTrain: 张量列类型
# sitedims: 获取每个张量的局部维度
# evaluate: 在指定索引处求值
export tensortrain, TensorTrain, sitedims, evaluate

# 张量收缩函数
export contract

# ===================================================================
# 包含子模块文件 (include)
# ===================================================================
# include语句将其他源文件的内容插入到当前位置
# 文件顺序很重要，因为后面的文件可能依赖前面定义的类型和函数

# 工具函数：通用辅助函数，如最大绝对值计算、随机子集选取等
include("util.jl")

# 扫描策略：定义TCI扫描方向的策略
include("sweepstrategies.jl")

# 抽象矩阵CI：矩阵交叉插值的抽象基类型
include("abstractmatrixci.jl")

# 矩阵CI：标准矩阵交叉插值实现
include("matrixci.jl")

# 矩阵ACA：自适应交叉近似(Adaptive Cross Approximation)算法
include("matrixaca.jl")

# 矩阵LU：秩揭示LU分解(rank-revealing LU decomposition)
include("matrixlu.jl")

# 矩阵LU-CI：基于LU分解的矩阵交叉插值
include("matrixluci.jl")

# 索引集：管理多重索引集合的数据结构
include("indexset.jl")

# 抽象张量列：张量列的抽象基类型，定义共同接口
include("abstracttensortrain.jl")

# 缓存张量列：带缓存的张量列求值，避免重复计算
include("cachedtensortrain.jl")

# 批量求值：支持批量函数求值的接口
include("batcheval.jl")

# 缓存函数：带记忆化的函数包装器，缓存已计算的值
include("cachedfunction.jl")

# 张量列：TensorTrain类型的完整实现
include("tensortrain.jl")

# TCI1：第一代张量交叉插值算法
include("tensorci1.jl")

# 全局枢轴搜索：寻找全局误差最大点的算法
include("globalpivotfinder.jl")

# TCI2：第二代张量交叉插值算法（改进版）
include("tensorci2.jl")

# 类型转换：不同表示之间的转换函数
include("conversion.jl")

# 积分：基于TCI的高维数值积分
include("integration.jl")

# 收缩：张量列之间的收缩运算
include("contraction.jl")

# 全局搜索：用于估计真实误差的全局搜索算法
include("globalsearch.jl")

end  # 模块结束
