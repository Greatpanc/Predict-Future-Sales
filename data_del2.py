# -*- encoding: utf-8 -*-

'''
Created on 2018年11月30日

@author: Greatpan
'''
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int8)
    return df

def date_del(data_path):
    import time
    start_time = time.time()

    day_sala_min=0
    day_sala_max=5

    import pandas as pd
    pd.set_option('display.max_rows', 99)        # 在控制台显示dataframe数据最多行数,超过后自动省略
    pd.set_option('display.max_columns', 50)     # 在控制台显示dataframe数据最多列数,超过后自动省略

    test = pd.read_csv('%s/test.csv' % data_path)
    
    all_set=test[['shop_id','item_id']].copy()
    all_set['date_block_num']=np.int8(12)
    for i in range(13,35):
        data_tmp=test[['shop_id','item_id']].copy()
        data_tmp['date_block_num']=np.int8(i)
        all_set=pd.concat([all_set,data_tmp],axis=0)
    
    item = pd.read_csv('%s/items.csv' % data_path)
    all_set = all_set.merge(item[['item_id', 'item_category_id']], on = ['item_id'],how = 'left')
    
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
    all_set = all_set.merge(item_cat[['item_cat_id_fix', 'item_category_id']], on = ['item_category_id'], how = 'left')
    
    date_columns = ['month', 'year', 'days_of_month']
    
    sale_train = pd.read_csv('%s/sales_train_v2.csv' % data_path)
    dates_train = sale_train[['date', 'date_block_num']].drop_duplicates()
    dates_train['date'] = pd.to_datetime(dates_train['date'], format = '%d.%m.%Y')
    dates_train['dow'] = dates_train['date'].dt.dayofweek    # 该天为星期几
    dates_train['year'] = dates_train['date'].dt.year        # 年份
    dates_train['month'] = dates_train['date'].dt.month.astype(np.int8)        # 份月
    dates_train = pd.get_dummies(dates_train, columns=['dow']) # 对'dow'数据重新进行离散编码,增加dow_0,dow_1,...,dow_6列代替原来的dow列
    dow_col = ['dow_' + str(x) for x in range(7)]
    date_features = dates_train.groupby(['year', 'month', 'date_block_num'])[dow_col].agg('sum').reset_index()
    date_features['days_of_month'] = date_features[dow_col].sum(axis=1).astype(np.int8)
    date_features['year'] = np.int8(date_features['year'] - 2013)
    date_features = date_features[[ 'year','month','days_of_month', 'date_block_num']]
    
    all_set=all_set.merge(date_features,on = 'date_block_num', how = 'left')
    data_tmp=all_set[all_set['date_block_num']==34]
    data_tmp['year']=np.int8(2)
    data_tmp['month']=np.int8(11)
    data_tmp['days_of_month']=np.int8(31)
    all_set[all_set['date_block_num']==34]=data_tmp
    
    Item_Index=['shop_id', 'item_id', 'date_block_num']
    sale_train['item_cnt_day']=sale_train['item_cnt_day'].clip(day_sala_min,day_sala_max)
    data_tmp=sale_train[['shop_id','item_id', 'date_block_num','item_cnt_day']]
    train_data=data_tmp.groupby(by=Item_Index)['item_cnt_day'].agg(['sum']).reset_index().rename(columns = {'sum': 'item_cnt_month'})
    train_data['item_cnt_month'] = train_data['item_cnt_month'].clip(0,40).astype(np.int8)
    train_data['on_sale']=1
    train_data = train_data.merge(item[['item_id', 'item_category_id']], on = ['item_id'], how = 'left')
    train_data = train_data.merge(item_cat[['item_cat_id_fix', 'item_category_id']], on = ['item_category_id'], how = 'left')
    
    Target = 'item_cnt_month'
    mean_encoded_col=[]

    from tqdm import tqdm
    for col in tqdm(['shop_id', 'item_id', 'item_category_id', 'item_cat_id_fix']):
        col_tr = train_data[['date_block_num']+[col]+[Target]]
        col_tr=col_tr.groupby(['date_block_num']+[col])[Target].agg('mean').reset_index().rename(columns ={Target:col+'_cnt_month_mean'})
        train_data=train_data.merge(col_tr,on=['date_block_num']+[col],how = 'left')
        mean_encoded_col.append(col+'_cnt_month_mean')

    # 2.3  提取当前月的前第一个月、前第二个月、前第三个月、前第四个月、前年该月的销量各特征作为特征
    id_col=['shop_id', 'item_id']
    index_cols = ['item_category_id', 'item_cat_id_fix', 'date_block_num']
    cols_to_rename = mean_encoded_col+[Target]
    print(cols_to_rename)
    shift_range = [1, 2, 3, 4, 12]        # 下一个月、两个月、三个月、四个月和下一年

    for month_shift in tqdm(shift_range):
        train_shift = train_data[id_col + index_cols + cols_to_rename].copy()
        train_shift['date_block_num'] = train_shift['date_block_num'] - month_shift
        foo = lambda x: '{}_pre_{}'.format(x, month_shift) if x in cols_to_rename else x
        train_shift = train_shift.rename(columns=foo)
        train_data = pd.merge(train_data, train_shift, on=id_col+index_cols, how='left').fillna(0)

    pre_cols = [col for col in train_data.columns if '_pre_' in col]
    all_set=all_set.merge(train_data[['shop_id','item_id','date_block_num','item_cnt_month']+pre_cols],on=['shop_id','item_id','date_block_num'], how = 'left').fillna(0)
    
    all_set = downcast_dtypes(all_set)

    # 2.4  对所有数据进行标准化处理
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    to_drop_cols = ['date_block_num']
    feature_columns = list(set(pre_cols + index_cols + list(all_set)).difference(to_drop_cols))
    all_set[feature_columns] = sc.fit_transform(all_set[feature_columns])

    ####################3.模型########################
    num_first_level_models=3
    meta_size=12
    slice_start = 0
    months_to_generate_meta_features=range(12+meta_size,35)       # 
    mask = all_set['date_block_num'].isin(months_to_generate_meta_features)
    y_all_level2 = all_set[Target][mask].values
    X_all_level2 = np.zeros([y_all_level2.shape[0], num_first_level_models])
    
    for cur_block_num in tqdm(months_to_generate_meta_features):
        # 3.0 建立训练集和测试集
        train_set=all_set[all_set['date_block_num']<cur_block_num][id_col+['item_category_id','item_cat_id_fix']+pre_cols+date_columns].values
        train_value=all_set[all_set['date_block_num']<cur_block_num]['item_cnt_month'].values
        test_set=all_set[all_set['date_block_num']==cur_block_num][id_col+['item_category_id','item_cat_id_fix']+pre_cols+date_columns].values
        test_value=all_set[all_set['date_block_num']==cur_block_num]['item_cnt_month']
        
        preds=[]
        # 3.1 GBDM模型
        import lightgbm as lgb
        lgb_params = {
                      'feature_fraction': 0.75,        # 每次迭代的时候随机选择特征的比例，默认为1，训练前选好
                      'metric': 'rmse',                # root square loss(平方根损失）
                      'nthread':1,                    # LightGBM 的线程数
                      'min_data_in_leaf': 2**7,        # 一个叶子上数据的最小数量. 可以用来处理过拟合
                      'bagging_fraction': 0.75,        # 类似于 feature_fraction, 但是它在训练时选特征
                      'learning_rate': 0.03,        # 学习率
                      'objective': 'mse',            # regression_l2, L2 loss, alias=regression, mean_squared_error, mse
                      'bagging_seed': 2**7,            # bagging 随机数种子
                      'num_leaves': 2**7,            # 一棵树上的叶子数
                      'bagging_freq':1,                # bagging 的频率, 0 意味着禁用 bagging. k意味着每 k次迭代执行bagging
                      'verbose':0                    # verbose: 详细信息模式，0 或者 1 
                      }
        estimator = lgb.train(lgb_params, lgb.Dataset(train_set, label=train_value), 300)
        pred_test = estimator.predict(test_set)
        pred_train = estimator.predict(train_set)
        preds.append(pred_test)
    
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        print('Train RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(train_value.clip(0,20), pred_train.clip(0,20)))))
        print('Test RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(test_value.clip(0,20).values, pred_test.clip(0,20)))))
    
        # 3.2 SGDRegressor模型
        from sklearn.linear_model import SGDRegressor
        sgdr= SGDRegressor(
            penalty = 'l2' ,            # 惩罚因子：L2正则化
            random_state = 0 
            )
        estimator=sgdr
        estimator.fit(train_set, train_value)            # 使用sgdr模型进行估计
        pred_test = estimator.predict(test_set)
        pred_train = estimator.predict(train_set)
        # preds.append(pred_test)
        print('Train RMSE for %s is %f' % ('SGDRegressor模型', sqrt(mean_squared_error(train_value.clip(0,20), pred_train.clip(0,20)))))
        print('Test RMSE for %s is %f' % ('SGDRegressor模型', sqrt(mean_squared_error(test_value.clip(0,20).values, pred_test.clip(0,20)))))
        
        # 3.3 KerasRegressor模型
        from keras.models import Sequential 
        from keras.layers import Dense
        from keras.wrappers.scikit_learn import KerasRegressor
        def baseline_model():
            # create model
            model = Sequential()
            model.add(Dense(20, input_dim=train_set.shape[1], kernel_initializer='uniform', activation='softplus'))
            model.add(Dense(1, kernel_initializer='uniform', activation = 'relu'))
            # Compile model
            model.compile(loss='mse', optimizer='Nadam', metrics=['mse'])
            # model.compile(loss='mean_squared_error', optimizer='adam')
            return model
        estimator = KerasRegressor(build_fn=baseline_model, verbose=0, epochs=5, batch_size = 55000)    # verbose: 详细信息模式，0 或者 1 
        estimator.fit(train_set, train_value)
        pred_test = estimator.predict(test_set)
        preds.append(pred_test)
        pred_train = estimator.predict(train_set)
        
        print('Train RMSE for %s is %f' % ('KerasRegressor模型', sqrt(mean_squared_error(train_value.clip(0,20), pred_train.clip(0,20)))))
        print('Test RMSE for %s is %f' % ('KerasRegressor模型', sqrt(mean_squared_error(test_value.clip(0,20).values, pred_test.clip(0,20)))))
        
        # 3.4 LinearRegression模型
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        estimator=lr
        estimator.fit(train_set, train_value)
        pred_test = estimator.predict(test_set)
        pred_train = estimator.predict(train_set)
        preds.append(pred_test)
        print('Train RMSE for %s is %f' % ('LinearRegression模型', sqrt(mean_squared_error(train_value.clip(0,20), pred_train.clip(0,20)))))
        print('Test RMSE for %s is %f' % ('LinearRegression模型', sqrt(mean_squared_error(test_value.clip(0,20).values, pred_test.clip(0,20)))))
        
        slice_end = slice_start + test_set.shape[0]
        X_all_level2[ slice_start : slice_end , :] = np.c_[preds].transpose()        # transpose用于转置的
        slice_start = slice_end
        print("运行时长为：%dmin%ds\t 完成一轮训练"%((time.time()-start_time)/60,(time.time()-start_time)%60))

    # 4. Ensembling -------------------------------------------------------------------
    test_nrow = len(preds[0])        # 预测的占的长度
    X_train_level2 = X_all_level2[ : -test_nrow, :]        # 训练集（前面预测的第27月到33月的值）test_nrow=214200
    X_test_level2 = X_all_level2[ -test_nrow: , :]        # 测试集（前面预测的第34月值）
    y_train_level2 = y_all_level2[ : -test_nrow]        # 训练集实际值
    # y_test_level2 = y_all_level2[ -test_nrow : ]        # 测试集实际值test_nrow = len(preds[0])        # 预测的占的长度
    
    # A. Second level learning model via linear regression第二层学习,模型1,使用线性回归模型
    lr = LinearRegression()
    lr.fit(X_train_level2, y_train_level2)
    test_preds_lr_stacking = lr.predict(X_test_level2)
    train_preds_lr_stacking = lr.predict(X_train_level2)
    print('Train R-squared for %s is %f' %('train_preds_lr_stacking', sqrt(mean_squared_error(y_train_level2, train_preds_lr_stacking))))
    
    submission_path="../Result"
    submission = pd.read_csv('%s/sample_submission.csv' % data_path)
    submission['item_cnt_month'] = test_preds_lr_stacking.clip(0,20)
    submission[['ID', 'item_cnt_month']].to_csv('%s/submission_Mode2_LinearRegression.csv' % (submission_path), index = False)
    
    print("运行时长为：%dmin%ds\t 预测完成"%((time.time()-start_time)/60,(time.time()-start_time)%60))
    print()

if __name__ == '__main__':
    import numpy as np
    
    import warnings
    warnings.filterwarnings('ignore')
    
    data_path = '../Data'
    date_del(data_path)
    
