import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

table = pq.read_table(r'd:\RynnScale\LLaVA\train-00000-of-00063.parquet')

# 关键：不要用 types_mapper！让 PyArrow 自动处理嵌套结构
df = table.to_pandas(
    ignore_metadata=True,   # 避免旧版 pandas metadata 导致的问题
    self_destruct=True,     # 节省内存（可选）
    use_threads=True
)

print(df.head())
print("\nConversations example:")
print(df['conversations'].iloc[0])