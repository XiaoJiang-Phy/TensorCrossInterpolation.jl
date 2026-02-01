# ===================================================================
# sweepstrategies.jl - 扫描策略模块
# ===================================================================
# 这个文件定义了TCI(张量交叉插值)算法中用于决定扫描方向的策略函数
# 
# 在TCI算法中，"扫描"是指按照一定顺序遍历张量链中的所有节点(sites)，
# 以更新插值参数。扫描可以是：
#   - 正向(forward): 从左到右
#   - 反向(backward): 从右到左  
#   - 往返(backandforth): 交替进行正向和反向扫描
# ===================================================================

"""
    forwardsweep(sweepstrategy::Symbol, iteration::Int) -> Bool

根据扫描策略和当前迭代次数，判断是否应该进行正向扫描。

# 参数
- `sweepstrategy::Symbol`: 扫描策略，可以是:
    - `:forward` - 始终正向扫描（从左到右）
    - `:backandforth` - 往返扫描（奇数次迭代正向，偶数次迭代反向）
- `iteration::Int`: 当前的迭代次数（从1开始计数）

# 返回值
- `Bool`: 如果应该正向扫描返回 `true`，否则返回 `false`

# 工作原理
- 如果策略是 `:forward`，总是返回 `true`（总是正向扫描）
- 如果策略是 `:backandforth`，在奇数次迭代时返回 `true`（正向），
  偶数次迭代时返回 `false`（反向）

# 示例
```julia
forwardsweep(:forward, 1)        # 返回 true
forwardsweep(:forward, 2)        # 返回 true
forwardsweep(:backandforth, 1)   # 返回 true (奇数次，正向)
forwardsweep(:backandforth, 2)   # 返回 false (偶数次，反向)
forwardsweep(:backandforth, 3)   # 返回 true (奇数次，正向)
```
"""
function forwardsweep(sweepstrategy::Symbol, iteration::Int)
    # 使用逻辑或(||)运算符:
    # - 条件1：如果策略是 :forward，直接返回 true
    # - 条件2：如果策略是 :backandforth 且迭代次数是奇数(isodd)，返回 true
    # - 其他情况返回 false，表示需要反向扫描
    return (
        (sweepstrategy == :forward) ||
        (sweepstrategy == :backandforth && isodd(iteration))
    )
end
