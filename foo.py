import torch
import gc
import numpy as np

# 创建张量并占用显存
x = torch.randn(10000, 10000, device='cuda')
x.cpu().numpy().tolist()

# 删除引用并触发垃圾回收
del x
# gc.collect()  # 清理 Python 对象（非必须，但建议）

# 强制释放框架管理的显存缓存
torch.cuda.empty_cache()

# 检查显存使用情况
print(torch.cuda.memory_allocated())  # 应为 0 或显著减少