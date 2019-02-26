import pandas as pd

df = pd.read_csv("./data/ads_train.csv", encoding="utf-8", header=0)
# 统计正负样本
print(df["y_buy"].value_counts())

# 空缺填0
df.fillna(0, inplace=True)

# 切分训练集、验证集
rate = 0.1
# df.drop_duplicates(keep='first', inplace=True)  # 去重
df = df.sample(frac=1.0)  # 全部打乱
cut_idx = int(round(0.1 * df.shape[0]))
df_valid, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
print (df.shape, df_valid.shape, df_train.shape)  
df_train.to_csv("./data/train.csv", float_format="%.5f", index=0)
df_valid.to_csv("./data/valid.csv", float_format="%.5f", index=0)

