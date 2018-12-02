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

	data_temp1=sale_train[['shop_id','item_id', 'date_block_num','item_cnt_day']]
	train_data=data_temp1.groupby(by=Item_Index)['item_cnt_day'].agg(['sum']).reset_index().rename(columns = {'sum': 'item_cnt_month'})
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
	# date_columns = ['month', 'year', 'days_of_month']
	print("汇入date_columns列后的train_data数据的数据量为：%d"%len(train_data))

	# 1.2 测试数据操作
	# 1.2.1  读取训练数据,并对训练集数据进行预处理
	sale_test  = pd.read_csv('%s/test.csv' % data_path)
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
	# 2.1  训练数据
	# 2.1.1  训练数据
	Target = 'item_cnt_month'
	mean_encoded_col=[]
	from tqdm import tqdm
	for col in tqdm(['shop_id', 'item_id', 'item_category_id', 'item_cat_id_fix']):
		col_tr = train_data[['date_block_num']+[col]+[Target]]
		col_tr=col_tr.groupby(['date_block_num']+[col])[Target].agg('mean').reset_index().rename(columns ={Target:col+'_cnt_month_mean'})
		train_data=train_data.merge(col_tr,on=['date_block_num']+[col],how = 'left')
		mean_encoded_col.append(col+'_cnt_month_mean')

	print(train_data.head())
	print(mean_encoded_col)

	# 2.1.2  训练数据
	index_cols = ['shop_id', 'item_id', 'item_category_id', 'item_cat_id_fix', 'date_block_num']
	cols_to_rename = mean_encoded_col+[Target]
	print(cols_to_rename)
	shift_range = [1, 2, 3, 4, 12]		# 下一个月、两个月、三个月、四个月和下一年
	
	for month_shift in tqdm(shift_range):
		train_shift = train_data[index_cols + cols_to_rename].copy()
		train_shift['date_block_num'] = train_shift['date_block_num'] - month_shift
		foo = lambda x: '{}_pre_{}'.format(x, month_shift) if x in cols_to_rename else x
		train_shift = train_shift.rename(columns=foo)
		train_data = pd.merge(train_data, train_shift, on=index_cols, how='left').fillna(0)


	train_data = train_data[train_data['date_block_num'] >= 12] # 不使用2013年的数据,因为缺失值多
	# lag_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]]
	train_data = downcast_dtypes(train_data)
	print('%0.2f min: Finish generating lag features'%((time.time() - start_time)/60))
		
if __name__ == '__main__':
	import numpy as np
	
	import warnings
	warnings.filterwarnings('ignore')
	
	data_path = '../Data'
	date_del(data_path)
	
