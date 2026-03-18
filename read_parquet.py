import pyarrow.parquet as pq

# 读取 Parquet 文件
table = pq.read_table(r"C:\Users\yunteng.li\Downloads\episode_007146.parquet")

# 1. 查看 Schema（列名 + 数据类型）
print("Schema:")
print(table.schema)

# 2. 查看前几行数据（可选）
print("\nFirst 5 rows:")
print(table.to_pandas().head())

# 3. 查看文件元信息（如行数、是否压缩等）
print("\nMetadata:")
print(f"Number of rows: {table.num_rows}")
print(f"Number of columns: {table.num_columns}")