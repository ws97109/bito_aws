import pandas as pd

# 读取标签文件获取user_id
print("读取 train_label.csv...")
train_label = pd.read_csv('RawData/train_label.csv')
train_user_ids = set(train_label['user_id'].unique())
print(f"训练集 user_id 数量: {len(train_user_ids)}")

print("\n读取 predict_label.csv...")
predict_label = pd.read_csv('RawData/predict_label.csv')
predict_user_ids = set(predict_label['user_id'].unique())
print(f"预测集 user_id 数量: {len(predict_user_ids)}")

# 检查是否有重叠
overlap = train_user_ids & predict_user_ids
if overlap:
    print(f"\n警告: 发现 {len(overlap)} 个重复的 user_id")
else:
    print("\n✓ 训练集和预测集的 user_id 没有重叠")

# 读取交易数据
print("\n读取 usdt_twd_trading.csv...")
trading_data = pd.read_csv('RawData/usdt_twd_trading.csv')
print(f"总交易记录数: {len(trading_data)}")
print(f"总用户数: {trading_data['user_id'].nunique()}")

# 根据user_id分割数据
print("\n开始分割数据...")
trading_train = trading_data[trading_data['user_id'].isin(train_user_ids)]
trading_predict = trading_data[trading_data['user_id'].isin(predict_user_ids)]

print(f"\n训练集交易记录: {len(trading_train)}")
print(f"训练集用户数: {trading_train['user_id'].nunique()}")

print(f"\n预测集交易记录: {len(trading_predict)}")
print(f"预测集用户数: {trading_predict['user_id'].nunique()}")

# 检查是否有user_id不在任何一个标签集中
all_label_ids = train_user_ids | predict_user_ids
trading_user_ids = set(trading_data['user_id'].unique())
missing_in_labels = trading_user_ids - all_label_ids
missing_in_trading_train = train_user_ids - trading_user_ids
missing_in_trading_predict = predict_user_ids - trading_user_ids

if missing_in_labels:
    print(f"\n注意: {len(missing_in_labels)} 个交易数据中的 user_id 不在标签集中")
if missing_in_trading_train:
    print(f"注意: {len(missing_in_trading_train)} 个训练标签的 user_id 在交易数据中没有记录")
if missing_in_trading_predict:
    print(f"注意: {len(missing_in_trading_predict)} 个预测标签的 user_id 在交易数据中没有记录")

# 保存分割后的数据
print("\n保存数据...")
trading_train.to_csv('RawData/usdt_twd_trading_train.csv', index=False)
print(f"✓ 已保存 usdt_twd_trading_train.csv")

trading_predict.to_csv('RawData/usdt_twd_trading_predict.csv', index=False)
print(f"✓ 已保存 usdt_twd_trading_predict.csv")

print("\n完成！")
