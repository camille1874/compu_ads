import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/ads_train.csv", encoding="utf-8", header=0)
# 查看数据信息
df.info()
df.describe()
df.duplicated(keep="last")
plt.figure()
df.plot.hist(grid=False, figsize=(12, 12))
plt.savefig('histogram.png')

print(df["y_buy"].value_counts())
# 0 38037
# 1 172
# 类别不平衡


# 预处理
df.fillna(0, inplace=True)
df["buy_freq"] = df["buy_freq"].astype(int)
df.drop_duplicates(keep='first', inplace=True) 


# 欠采样
df0 = df[df["y_buy"] == 0]
df1 = df[df["y_buy"] == 1]
df2 = df0.sample(frac=0.1)
df = pd.concat([df1, df2], ignore_index=True)


# 切分训练集、验证集
rate = 0.3
df = df.sample(frac=1.0)  # 全部打乱
cut_idx = int(round(rate * df.shape[0]))
df_valid, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]


print (df.shape, df_valid.shape, df_train.shape)  
df_train.to_csv("./data/train.csv", float_format="%.5f", index=0)
df_valid.to_csv("./data/valid.csv", float_format="%.5f", index=0)

