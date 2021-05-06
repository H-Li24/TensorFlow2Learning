import tensorflow as tf
import pandas as pd

df = pd.read_csv('heart.csv')

print(df.head())
print(df.dtypes)

# 把object格式数据转换成int8的顺序变量
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes

print('\n')

print(df.head())
print(df.dtypes)

print('\n')

target = df.pop('target') # 取target这一列赋给target并在df中删除target这一列
print(target)
print(df.values) # 取dataframe的值构成numpy array
print(target.values)

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
for feature, target in dataset.take(5):
    print('Feature:{}, Target:{}'.format(feature, target))