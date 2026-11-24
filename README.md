# pairwise-rendergraph-alias-barrier

## 领域

游戏、图形与多媒体

## 主题

渲染图资源别名与屏障

## 说明

跨帧引用、间接写入与跨队列屏障。

初始状态下 `graphics/rendergraph_alias_barrier.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
