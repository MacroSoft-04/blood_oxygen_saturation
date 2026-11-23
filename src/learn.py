import pandas as pd

tr = pd.read_parquet("train_input.parquet")
tgt = pd.read_csv("train_target.csv")
meta = pd.read_csv("case_metadata.csv")

# +10분 MBP 정답만
y10 = tgt[(tgt.variable == "MBP") & (tgt.horizon == 10)]

# 한 윈도우 보기
tr[tr.sample_id == tr.sample_id.iloc[0]]
