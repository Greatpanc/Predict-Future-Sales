# -*- encoding: utf-8 -*-

'''
Created on 2018年11月30日

@author: Greatpan
'''
def downcast_dtypes(df):
	float_cols = [c for c in df if df[c].dtype == "float64"]
	int_cols =   [c for c in df if df[c].dtype in ["int64", "int32"]]
	df[float_cols] = df[float_cols].astype(np.float32)
	df[int_cols]   = df[int_cols].astype(np.int16)
	return df

def date_del(data_path):
	import time
	start_time = time.time()

	import pandas as pd
	pd.set_option('display.max_rows', 99)		# 在控制台显示dataframe数据最多行数,超过后自动省略
	pd.set_option('display.max_columns', 50)	# 在控制台显示dataframe数据最多列数,超过后自动省略
	
	####################1.读取文件########################
	# 1.1  训练数据
	# 1.1.1  读取训练数据,并对训练集数据进行预处理
	sale_train = pd.read_csv('%s/sales_train_v2.csv' % data_path)
	print("运行时长为：%dmin%ds\t sale_train数据的数据量为：%d"%((time.time()-start_time)/60,(time.time()-start_time)%60,len(sale_train)))
	
	# 1.1.2 计算不同ID下（shop_id、item_id）下的月销量存放在"item_cnt_month"下
	Item_Index=['shop_id', 'item_id', 'date_block_num']

	data_tmp=sale_train[['shop_id','item_id', 'date_block_num','item_cnt_day']]
	train_data=data_tmp.groupby(by=Item_Index)['item_cnt_day'].agg(['sum']).reset_index().rename(columns = {'sum': 'item_cnt_month'})
	train_data['item_cnt_month'] = train_data['item_cnt_month'].astype(int).fillna(0)
	print("汇入item_cnt_month列后的train_data数据的数据量为：%d"%len(train_data))

	# 1.1.3  将每个item_id对应的item_categroy_id属性汇入训练数据集中
	item = pd.read_csv('%s/items.csv' % data_path)
	train_data = train_data.merge(item[['item_id', 'item_category_id']], on = ['item_id'], how = 'left')
	print("汇入item_categroy_id列后的train_data数据的数据量为：%d"%len(train_data))

	# 1.1.4 通过商品类别名对类别进行修正,使同一个类别不同信号归为同一个类别,并将修正后的item_cat_id_fix属性汇入训练数据集中
	item_cat = pd.read_csv('%s/item_categories.csv' % data_path)
	item_cat.item_category_name[0]=1
	item_cat.item_category_name[1:8]=2
	item_cat.item_category_name[8]=3
	item_cat.item_category_name[9]=4
	item_cat.item_category_name[10:18]=5
	item_cat.item_category_name[18:25]=6
	item_cat.item_category_name[25]=7
	item_cat.item_category_name[26:28]=8
	item_cat.item_category_name[28:32]=9
	item_cat.item_category_name[32:37]=10
	item_cat.item_category_name[37:43]=11
	item_cat.item_category_name[43:55]=12
	item_cat.item_category_name[55:61]=13
	item_cat.item_category_name[61:73]=14
	item_cat.item_category_name[73:79]=15
	item_cat.item_category_name[79:81]=16
	item_cat.item_category_name[81:83]=17
	item_cat.item_category_name[83]=18
	item_cat=item_cat.rename(columns = {'item_category_name': 'item_cat_id_fix'})
	item_cat['item_cat_id_fix']=item_cat['item_cat_id_fix'].astype(np.int16)
	train_data = train_data.merge(item_cat[['item_cat_id_fix', 'item_category_id']], on = ['item_category_id'], how = 'left')
	print("汇入item_cat_id_fix列后的train_data数据的数据量为：%d"%len(train_data))

	# 1.1.5 将日期特征['month', 'year', 'days_of_month']汇入到训练集当中
	dates_train = sale_train[['date', 'date_block_num']].drop_duplicates()
	dates_train['date'] = pd.to_datetime(dates_train['date'], format = '%d.%m.%Y')
	dates_train['dow'] = dates_train['date'].dt.dayofweek	# 该天为星期几
	dates_train['year'] = dates_train['date'].dt.year		# 年份
	dates_train['month'] = dates_train['date'].dt.month		# 份月
	dates_train = pd.get_dummies(dates_train, columns=['dow']) # 对'dow'数据重新进行离散编码,增加dow_0,dow_1,...,dow_6列代替原来的dow列
	dow_col = ['dow_' + str(x) for x in range(7)]
	date_features = dates_train.groupby(['year', 'month', 'date_block_num'])[dow_col].agg('sum').reset_index()
	date_features['days_of_month'] = date_features[dow_col].sum(axis=1)
	date_features['year'] = date_features['year'] - 2013
	date_features = date_features[[ 'year','month','days_of_month', 'date_block_num']]
	train_data = train_data.merge(date_features, on = 'date_block_num', how = 'left')
	# train_data[['month','days_of_month']].drop_duplicates().sort_values(by='month')	# 检查数据完整性
	date_columns = ['month', 'year', 'days_of_month']
	print("汇入date_columns列后的train_data数据的数据量为：%d"%len(train_data))

	# 1.2 测试数据操作
	# 1.2.1  读取训练数据,并对训练集数据进行预处理
	test = pd.read_csv('%s/test.csv' % data_path)
	sale_test  = test.copy()
	print("sale_test数据的数据量为：%d"%len(sale_test))
	
	# 1.2.2  将测试数据汇入测试集当中
	test_data=sale_test[['shop_id', 'item_id']]
	test_data['date_block_num']=34
	print("test_train数据的数据量为：%d"%len(test_data))
	
	# 1.2.3 将每个item_id对应的item_categroy_id属性汇入训练数据集中
	test_data = test_data.merge(item[['item_id', 'item_category_id']], on = ['item_id'], how = 'left')
	print("汇入item_categroy_id列后的test_data数据的数据量为：%d"%len(test_data))

	# 1.2.4 通过商品类别名对类别进行修正,使同一个类别不同信号归为同一个类别,并将修正后的item_cat_id_fix属性汇入训练数据集中
	test_data = test_data.merge(item_cat[['item_cat_id_fix', 'item_category_id']], on = ['item_category_id'], how = 'left')
	print("汇入item_cat_id_fix列后的test_data数据的数据量为：%d"%len(test_data))
	
	# 1.2.5 将日期特征['month', 'year', 'days_of_month']汇入到测试集test_data当中
	test_data['year']=2
	test_data['month']=11
	test_data['days_of_month']=31
	print("汇入date_columns列后的test_data数据的数据量为：%d"%len(train_data))
	train_data=downcast_dtypes(train_data)


	####################2.特征工程########################
	# 2.1 建立数据集
	all_data=pd.concat([train_data,test_data],axis=0)

	# 2.2  分别计算不同shop、item_id、item_category_id、item_cat_id_fix的每月商品销量的平均值作为特征
	Target = 'item_cnt_month'
	mean_encoded_col=[]

	from tqdm import tqdm
	for col in tqdm(['shop_id', 'item_id', 'item_category_id', 'item_cat_id_fix']):
		col_tr = all_data[['date_block_num']+[col]+[Target]]
		col_tr=col_tr.groupby(['date_block_num']+[col])[Target].agg('mean').reset_index().rename(columns ={Target:col+'_cnt_month_mean'})
		all_data=all_data.merge(col_tr,on=['date_block_num']+[col],how = 'left')
		mean_encoded_col.append(col+'_cnt_month_mean')

	print(mean_encoded_col)

	# 2.3  提取当前月的前第一个月、前第二个月、前第三个月、前第四个月、前年该月的销量各特征作为特征
	id_col=['shop_id', 'item_id']
	index_cols = ['item_category_id', 'item_cat_id_fix', 'date_block_num']
	cols_to_rename = mean_encoded_col+[Target]
	print(cols_to_rename)
	shift_range = [1, 2, 3, 4, 12]		# 下一个月、两个月、三个月、四个月和下一年

	for month_shift in tqdm(shift_range):
		train_shift = all_data[id_col + index_cols + cols_to_rename].copy()
		train_shift['date_block_num'] = train_shift['date_block_num'] - month_shift
		foo = lambda x: '{}_pre_{}'.format(x, month_shift) if x in cols_to_rename else x
		train_shift = train_shift.rename(columns=foo)
		all_data = pd.merge(all_data, train_shift, on=id_col+index_cols, how='left').fillna(0)

	all_data = all_data[all_data['date_block_num'] >= 12] # 不使用2013年的数据,因为缺失值多
	pre_cols = [col for col in all_data.columns if '_pre_' in col]
	all_data = downcast_dtypes(all_data)

	# 2.4  对所有数据进行标准化处理
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	to_drop_cols = ['date_block_num']
	feature_columns = list(set(pre_cols + index_cols + list(date_columns)).difference(to_drop_cols))
	all_data[feature_columns] = sc.fit_transform(all_data[feature_columns])

	# 2.5 建立训练集和测试集
	all_set=test[['shop_id','item_id']].copy()
	all_set['date_block_num']=np.int8(12)
	for i in range(13,35):
		data_tmp=sale_test[['shop_id','item_id']].copy()
		data_tmp['date_block_num']=np.int8(i)
		all_set=pd.concat([all_set,data_tmp],axis=0)
		
	all_set=all_set.merge(all_data,on=['shop_id','item_id','date_block_num'],how='left').fillna(0)
	all_set[id_col] = sc.fit_transform(all_set[id_col])
	
	train_set=all_set[all_set['date_block_num']<34][id_col+['item_category_id','item_cat_id_fix']+pre_cols+date_columns]
	train_value=all_set[all_set['date_block_num']<34]['item_cnt_month']
	test_set=all_set[all_set['date_block_num']==34][id_col+['item_category_id','item_cat_id_fix']+pre_cols+date_columns]
# 	test_value=all_set[all_set['date_block_num']==34]['item_cnt_month']
	
	####################3.模型########################
	# 2.1 建立模型
	import lightgbm as lgb
	lgb_params = {
                  'feature_fraction': 0.75,		# 每次迭代的时候随机选择特征的比例，默认为1，训练前选好
                  'metric': 'rmse',				# root square loss(平方根损失）
                  'nthread':1,					# LightGBM 的线程数
                  'min_data_in_leaf': 2**7,		# 一个叶子上数据的最小数量. 可以用来处理过拟合
                  'bagging_fraction': 0.75,		# 类似于 feature_fraction, 但是它在训练时选特征
                  'learning_rate': 0.03,		# 学习率
                  'objective': 'mse',			# regression_l2, L2 loss, alias=regression, mean_squared_error, mse
                  'bagging_seed': 2**7,			# bagging 随机数种子
                  'num_leaves': 2**7,			# 一棵树上的叶子数
                  'bagging_freq':1,				# bagging 的频率, 0 意味着禁用 bagging. k意味着每 k次迭代执行bagging
                  'verbose':1					# verbose: 详细信息模式，0 或者 1 
                  }
	estimator = lgb.train(lgb_params, lgb.Dataset(train_set.values, label=train_value.values), 300)
	pred_test = estimator.predict(test_set.values)
	pred_train = estimator.predict(train_set.values)

	from sklearn.metrics import mean_squared_error
	from math import sqrt
	print('Train RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(train_value.clip(0,20).values, pred_train.clip(0,20)))))
# 	print('Test RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(test_value.clip(0,20).values, pred_test.clip(0,20)))))

	submission_path="."
	submission = pd.read_csv('%s/sample_submission.csv' % data_path)
	submission['item_cnt_month'] = pred_test.clip(0,20)
	submission[['ID', 'item_cnt_month']].to_csv('%s/submission.csv' % (submission_path), index = False)
	
	print("运行时长为：%dmin%ds\t 特征提取结束"%((time.time()-start_time)/60,(time.time()-start_time)%60))
	
if __name__ == '__main__':
	import numpy as np
	
	import warnings
	warnings.filterwarnings('ignore')
	
	data_path = '../Data'
	date_del(data_path)
	
