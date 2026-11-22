# render-deferred-lighting

延迟渲染的 G-Buffer 编码 / 重建小练习：8 个待实现函数，15 条行为测试，初始状态全红。

## 题面

改成延迟着色后，背光面也发亮、多光源时环境光被叠了好几遍：encode_normal 的八面体折叠只处理上半球，z<0 的法线被编码到错误位置，背光方向整体偏掉；decode_normal 反解后没有归一化，单位性丢失，光照偏亮；encode_depth 直接存线性深度，远处精度不足，重建位置抖动；reconstruct_pos 不做 w 除法，位置偏差随深度线性增大；shade_gbuffer 对每个光源都把环境光加一次，N 个光源时环境光被叠 N 倍；shade_gbuffer 的衰减用 1/d，远处过亮；resolve_msaa 的深度取平均，边缘出现假轮廓；validate_gbuffer 不检查法线单位长度，非单位法线未被报出；light_batch 复用 G-Buffer 引用，第二帧还在用第一帧的法线；shade_gbuffer 在半透明像素上也写深度，后续剔除错误。deferred/gbuffer.py 是核心，tests/test_gbuffer.py 15个红。要修的是：八面体编解码严格互逆并归一化、1/z 非线性深度与逆投影 w 除法、环境光只加一次、1/d² 衰减、MSAA 深度取最近、法线单位校验、批量帧独立、半透明不写深度。难点是编码与重建：八面体折叠、深度非线性、w 除法三处必须严格互逆。

## 运行方式

仓库根目录执行（Python 3 + pytest，离线、无外部依赖）：

```bash
python -m pytest tests -q
```

初始状态 8 个函数全部 `raise NotImplementedError`，因此恰好 15 个失败，且不应出现
collection / import / syntax error：

```text
15 failed in 0.xxs
```

实现 `deferred/gbuffer.py` 的 8 个函数后，同一命令应变为 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 只允许使用标准库（`math`）与 pytest；离线、无随机数。
- 只允许修改 `deferred/gbuffer.py`；**`tests/test_gbuffer.py` 不可修改**（15 条测试是验收标准）。
- `conftest.py` 只负责把仓库根目录加入 `sys.path`。
- 接口契约以 `deferred/gbuffer.py` 的 docstring 为准：深度归一化到 0..1、法线为单位向量、数值容差 1e-9。
