# -*- encoding: utf-8 -*-

'''
Created on 2018年11月30日

@author: Greatpan
'''
from numpy import NaN

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int16)
    return df

def lag_feature(df, shift_range, cols_to_rename):
    for month_shift in tqdm(shift_range):
        train_shift = df[id_cols +[date_block_col]+ cols_to_rename].copy()
        train_shift['date_block_num'] = train_shift['date_block_num'] - month_shift
        foo = lambda x: '{}_pre_{}'.format(x, month_shift) if x in cols_to_rename else x
        train_shift = train_shift.rename(columns=foo)
        df = df.merge(train_shift, on=id_cols+[date_block_col], how='left')
    return df

# ************************************数据清洗与集成*****************************
def data_cleansing_integration():
    day_sala_min=0
    day_sala_max=20

    # 读取销售记录文件
    sale_train = pd.read_csv('%s/sales_train_v2.csv' % data_path)

    # 对异常数据修正(商店重了错误)
    sale_train.loc[sale_train.shop_id == 0,'shop_id'] = 57
    sale_train.loc[sale_train.shop_id == 1,'shop_id'] = 58
    sale_train.loc[sale_train.shop_id ==11,'shop_id'] = 10
    
    # 对异常数据修正(商店已经不营业了,但存在退货记录,直接剔除)
    sale_train=sale_train.loc[~((sale_train.shop_id== 8) & (sale_train.date_block_num==3))]
    sale_train=sale_train.loc[~((sale_train.shop_id==27) & (sale_train.date_block_num==32))]
    sale_train=sale_train.loc[~((sale_train.shop_id==33) & (sale_train.date_block_num==27))]
    sale_train=sale_train.loc[~((sale_train.shop_id==34) & (sale_train.date_block_num==18))]

    # 剔除退货商品（item_cnt_day<-1）
    sale_train=sale_train.loc[sale_train.item_cnt_day>0]

    # 对异常数据修正(剔除价格异常的数据)
    sale_train.loc[sale_train.item_id==11365,'item_price']=sale_train.loc[sale_train.item_id==11365].item_price.clip(0,6000)
    sale_train.loc[(sale_train.item_id==2973)&(sale_train.item_price==-1),'item_price']=1249
    sale_train.loc[(sale_train.item_id==8540)&(sale_train.item_price==1 ),'item_price']=299
    sale_train = sale_train[sale_train.item_price<100000]
    sale_train=sale_train.loc[~((sale_train.shop_id==32) & (sale_train.item_id==2973))]

    # 建立训练数据集
    grid = []
    from itertools import product
    for block_num in sale_train['date_block_num'].unique():
        cur_shops = sale_train[sale_train['date_block_num']==block_num]['shop_id'].unique()
        cur_items = sale_train[sale_train['date_block_num']==block_num]['item_id'].unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
    
    index_cols = ['shop_id', 'item_id', 'date_block_num']
    train_set = pd.DataFrame(np.vstack(grid),columns=index_cols,dtype=np.int32)
    del grid
    
    # 读取测试集文件
    test = pd.read_csv('%s/test.csv' % data_path)
    test_set = test[['shop_id','item_id']].copy()
    test_set['date_block_num']=34
    
    all_set=pd.concat([train_set,test_set])
    all_set.shop_id=all_set.shop_id.astype(np.int8)
    all_set.date_block_num=all_set.date_block_num.astype(np.int8)
    
    # 求取商店城市信息
    from sklearn.preprocessing import LabelEncoder
    shops = pd.read_csv('%s/shops.csv' % data_path)
    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
    shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
    shops = shops[['shop_id','city_code']]
    all_set=all_set.merge(shops, on = 'shop_id', how = 'left')
    all_set['city_code']=all_set['city_code'].astype(np.int8)
    
    # 将item所属类别汇入数据集网格中(数据文件给出的类别)
    item = pd.read_csv('%s/items.csv' % data_path)
    all_set = all_set.merge(item[['item_id', 'item_category_id']], on = ['item_id'],how = 'left')
    all_set.item_id = all_set.item_id.astype(np.int16)
    all_set.item_category_id = all_set.item_category_id.astype(np.int8)
    
    # 商品的类别特征（由名字划分的类别与子类别）
    cats = pd.read_csv('%s/item_categories.csv'% data_path)
    cats['split']= cats['item_category_name'].str.split('-')
    cats['type'] = cats['split'].map(lambda x: x[0].strip())
    cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
    
    cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    cats['subtype'] = cats['subtype'].map(lambda x: x.split('(')[0].strip())
    cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
    cats = cats[['item_category_id','type_code', 'subtype_code']]
    
    all_set=all_set.merge(cats, on = 'item_category_id', how = 'left')
    all_set['type_code']=all_set['type_code'].astype(np.int8)
    all_set['subtype_code']=all_set['subtype_code'].astype(np.int8)
    del shops,item,cats
    
    # 对sale_data文件中记录的月销售数据和产品月平均售价进行特征提取（'item_cnt_month')
    id_cols=['shop_id', 'item_id']
    sale_train['item_cnt_day']=sale_train['item_cnt_day'].clip(day_sala_min,day_sala_max)
    data_tmp=sale_train[id_cols+[date_block_col]+['item_cnt_day']]
    data_tmp=data_tmp.groupby(by=id_cols+[date_block_col])['item_cnt_day'].agg(['sum']).reset_index().rename(columns={'sum':'item_cnt_month'})
    
    data_tmp['item_cnt_month'] = data_tmp['item_cnt_month'].clip(0,20).astype(np.int8)
    all_set=all_set.merge(data_tmp,on=['shop_id','item_id','date_block_num'],how='left').fillna(0)
    
    # 提取商品的价格均值特征
    data_tmp = sale_train.groupby(['item_id']).agg({'item_price': ['mean']})
    data_tmp.columns = ['item_price_mean']
    data_tmp.reset_index(inplace=True)
    
    all_set = pd.merge(all_set, data_tmp, on=['item_id'], how='left')
    all_set['item_price_mean'] = all_set['item_price_mean'].astype(np.float16)
    
    # 提取商品价格月均值特征
    data_tmp = sale_train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
    data_tmp.columns = ['item_price_mean_month']
    data_tmp.reset_index(inplace=True)
    
    all_set = pd.merge(all_set, data_tmp, on=['date_block_num','item_id'], how='left')
    all_set['item_price_mean_month'] = all_set['item_price_mean_month'].astype(np.float16)
    
    # 求价格相对平均价格的浮动值, 偏置月份为[1,2,3,4,5,6]
    lags = [1,2,3,4,5,6]
    all_set = lag_feature(all_set, lags, ['item_price_mean_month'])
    for i in lags:  
        all_set['item_price_delta_month_pre_'+str(i)] = \
            (all_set['item_price_mean_month_pre_'+str(i)] - all_set['item_price_mean']) / all_set['item_price_mean']
    
    #求最近的价格浮动值（如果上一个月内该产品没有被销售,则用上上一个月的价格浮动值代替）
    all_set['item_price_delta_month']=all_set['item_price_delta_month_pre_1'].copy()
    for i in [2,3,4,5]:   
        all_set.loc[all_set['item_price_delta_month'].isna(),'item_price_delta_month'] \
            =all_set.loc[all_set['item_price_delta_month'].isna(),'item_price_delta_month_pre_'+str(i)]
    all_set=all_set.fillna(0)
    
    # 提取营业额特征['revenue']=['item_price']*['item_cnt_day']
    sale_train['revenue']=sale_train['item_price']*sale_train['item_cnt_day'] 
    group = sale_train.groupby(['date_block_num','shop_id']).agg({'revenue':['sum']})
    group.columns = ['shop_id_revenue_month']
    group.reset_index(inplace=True)
    
    all_set = pd.merge(all_set, group, on=['date_block_num','shop_id'], how='left')
    all_set['shop_id_revenue_month'] = all_set['shop_id_revenue_month'].astype(np.float32)
    
    group = group.groupby(['shop_id']).agg({'shop_id_revenue_month': ['mean']})
    group.columns = ['shop_id_revenue']
    group.reset_index(inplace=True)
    
    all_set = pd.merge(all_set, group, on=['shop_id'], how='left')
    all_set['shop_id_revenue'] = all_set['shop_id_revenue'].astype(np.float32)
    
    all_set['revenue_float_month'] = (all_set['shop_id_revenue_month'] - all_set['shop_id_revenue']) / all_set['shop_id_revenue']
    all_set['revenue_float_month'] = all_set['revenue_float_month'].astype(np.float16)
    all_set = lag_feature(all_set,[1], ['revenue_float_month'])
    all_set.drop(['shop_id_revenue','shop_id_revenue_month','revenue_float_month'], axis=1, inplace=True)
    
    # 提取日期特征('month', 'year','days_of_month')
    sale_train['date'] = pd.to_datetime(sale_train['date'], format = '%d.%m.%Y')
    data_tmp=sale_train[['shop_id','date', 'date_block_num']].drop_duplicates()

    data_tmp['day']=data_tmp['date'].dt.day
    data_tmp=data_tmp[['shop_id','date_block_num','day']]
    data_tmp=data_tmp.groupby(['shop_id','date_block_num']).agg(['min','max']).reset_index()
    data_tmp['days_of_month']=data_tmp['day','max']+1-data_tmp['day','min']
    data_tmp=data_tmp[['shop_id','date_block_num','days_of_month']]
    data_tmp.columns=data_tmp.columns.droplevel(1)
    
    all_set=all_set.merge(data_tmp,on=['shop_id','date_block_num'],how='left')
    all_set.loc[all_set.date_block_num==34,'days_of_month']=30
    all_set.days_of_month=all_set.days_of_month.astype(np.int8)
    
    all_set['month']=all_set[date_block_col]%12
    all_set['month']=all_set['month'].astype(np.int8)
    
    all_set['year']=(all_set[date_block_col]/12)
    all_set['year']=all_set['year'].astype(np.int8)
    
    return all_set

# ************************************ 特征工程 *****************************
def data_feature_extract(all_set):
    train_data = all_set[all_set['date_block_num']<=33]
    
    # 提取shop_code特征
    data_tmp=train_data.groupby(by=['shop_id','date_block_num'])['item_cnt_month'].sum().reset_index()
    data_tmp=data_tmp.groupby(by=['shop_id'])['item_cnt_month'].mean().reset_index().rename(columns={'item_cnt_month':'shop_code'})
    data_tmp['shop_code']=((data_tmp.shop_code/50)+0.5).astype(np.int16).clip(0,200).map(lambda x:x if x<=70 else 70+(x-70)/10).astype(np.int16)# 70这个值是由盒形图决定取的~
    train_data=train_data.merge(data_tmp,on=['shop_id'],how='left')
    all_set=all_set.merge(data_tmp,on=['shop_id'],how='left')
    
    # 提取item_category_code特征
    data_tmp=train_data.groupby(by=['item_category_id','date_block_num'])['item_cnt_month'].sum().reset_index()
    data_tmp=data_tmp.groupby(by=['item_category_id'])['item_cnt_month'].mean().reset_index().rename(columns={'item_cnt_month':'item_category_code'})
    data_tmp['item_category_code']=(data_tmp.item_category_code.clip(0,2000)/10+0.5).map(lambda x:x if x<=100 else 100+(x-100)/10).astype(np.int16)
    train_data=train_data.merge(data_tmp,on=['item_category_id'],how='left')
    all_set=all_set.merge(data_tmp,on=['item_category_id'],how='left')
    
    # 提取item_code特征
    data_tmp=train_data.groupby(by=['item_id','date_block_num'])['item_cnt_month'].sum().reset_index()
    data_tmp=data_tmp.groupby(by=['item_id'])['item_cnt_month'].mean().reset_index().rename(columns={'item_cnt_month':'item_code'})
    data_tmp['item_code']=(data_tmp['item_code']+0.5).clip(0,200).astype(np.int16).map(lambda x:x if x<=25 else 25+(x-25)/10).astype(np.int16)
    train_data=train_data.merge(data_tmp,on=['item_id'],how='left')
    all_set=all_set.merge(data_tmp,on=['item_id'],how='left')
    
    # 用['item_id','type_code','subtype_code']下的月均值对['item_code']缺失项进行补足
    data_tmp=all_set.loc[all_set.date_block_num<34,['item_id','type_code','subtype_code','item_code']]
    
    data_tmp=data_tmp.groupby(['type_code','subtype_code'])['item_code'].median().reset_index().rename(columns={'item_code':'item_code_fix'})
    all_set.loc[all_set.item_code.isna(),'item_code']=all_set.loc[all_set.item_code.isna()]\
        .merge(data_tmp,on=['type_code','subtype_code'],how='left').item_code_fix.values
    
    # (shop_id、item_id)、(item_id)中上一次销售出去的月份
    # shop_item_last_sale=pd.DataFrame()
    # item_id_last_sale=pd.DataFrame()
    all_set['shop_item_last_sale'] = NaN
    all_set['item_id_last_sale'] = NaN
    for i in range(0,35):
        data_tmp=all_set.loc[(all_set.date_block_num<i)&(all_set.item_cnt_month>0),['shop_id','item_id','date_block_num']].drop_duplicates()
        data_tmp=data_tmp.groupby(by=['shop_id','item_id'])['date_block_num'].max().reset_index().rename(columns={'date_block_num':'shop_item_last_sale'})
        data_tmp['shop_item_last_sale']=i-data_tmp['shop_item_last_sale']
        data_tmp=all_set.loc[all_set.date_block_num==i,['shop_id','item_id']].merge(data_tmp,on=['shop_id','item_id'],how='left')
        # shop_item_last_sale=pd.concat([shop_item_last_sale, data_tmp['shop_item_last_sale']], axis = 0)
        all_set.loc[all_set.date_block_num==i,'shop_item_last_sale']=data_tmp['shop_item_last_sale'].values
        
        data_tmp=all_set.loc[(all_set.date_block_num<i)&(all_set.item_cnt_month>0),['item_id','date_block_num']].drop_duplicates()
        data_tmp=data_tmp.groupby(by=['item_id'])['date_block_num'].max().reset_index().rename(columns={'date_block_num':'item_id_last_sale'})
        data_tmp['item_id_last_sale']=i-data_tmp['item_id_last_sale']
        data_tmp=all_set.loc[all_set.date_block_num==i,['item_id']].merge(data_tmp,on=['item_id'],how='left')
        # item_id_last_sale=pd.concat([item_id_last_sale, data_tmp['item_id_last_sale']], axis = 0)
        all_set.loc[all_set.date_block_num==i,'item_id_last_sale']=data_tmp['item_id_last_sale'].values
    
    
    #all_set['shop_item_last_sale'] = shop_item_last_sale[0]
    all_set['shop_item_last_sale'] = all_set['shop_item_last_sale'].fillna(-1)
    all_set['shop_item_last_sale'] = all_set['shop_item_last_sale'].astype(np.int8)
    
    # all_set=pd.concat([all_set, item_id_last_sale], axis = 1)
    
    #all_set['item_id_last_sale'] = item_id_last_sale[0]
    all_set['item_id_last_sale'] = all_set['item_id_last_sale'].fillna(-1)
    all_set['item_id_last_sale'] = all_set['item_id_last_sale'].astype(np.int8)  
    
    # shop_id、item_id下的第一次销售月份距离当前月份值
    all_set['shop_item_first_sale'] = all_set['date_block_num'] - all_set.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
    all_set['item_id_first_sale'] = all_set['date_block_num'] - all_set.groupby('item_id')['date_block_num'].transform('min')
    
    # 求['shop_id','item_category_id','item_id','city_code','type_code','subtype_code']下各均值特征
    global_mean =  train_data['item_cnt_month'].mean()
    corrcoefs = pd.DataFrame(columns = ['Cor'])
    y_tr = train_data['item_cnt_month'].values
    colofCode=['shop_id','item_category_id','item_id','city_code','type_code','subtype_code']
    for col in colofCode:
        col_tr=train_data[[col]+['item_cnt_month']]
        cumsum = col_tr.groupby(col)['item_cnt_month'].cumsum() - col_tr['item_cnt_month']
        sumcnt = col_tr.groupby(col).cumcount()+1
        col_tr[col + '_cnt_month_mean'] = cumsum / sumcnt
        col_tr[col + '_cnt_month_mean'].fillna(global_mean, inplace=True)
        corrcoefs.loc[col + '_cnt_month_mean'] = np.corrcoef(y_tr, col_tr[col + '_cnt_month_mean'])[0][1]
        train_data = pd.concat([train_data, col_tr[col + '_cnt_month_mean']], axis = 1)
        print(corrcoefs.sort_values('Cor'))
    
    train_data=downcast_dtypes(train_data)
    feature_col=[col for col in train_data.columns if '_cnt_month_mean' in col]
    
    all_set=all_set.merge(train_data[id_cols+[date_block_col]+feature_col],on=id_cols+[date_block_col], how = 'left')
    
    del data_tmp

    # 提取当前月的前第一个月、前第二个月、前第三个月、前第四个月、前年该月的销量各特征作为特征
    #shift_range = [1,2,3,6,12]        # 下一个月、两个月、三个月、四个月和下一年
    
    
    all_set=all_set[all_set.date_block_num>=12]
    # 求['shop_id_cnt_month_mean'、'item_id_cnt_month_mean'、'item_cnt_month']的[1,2,3,6,12]偏置
    cols_to_rename=[
                    'shop_id_cnt_month_mean',
                    'item_id_cnt_month_mean',
                    'item_cnt_month'
                    ]
    shift_range = [1,2,3,6,12]
    all_set=lag_feature(all_set,shift_range,cols_to_rename)
    # 求['shop_id_cnt_month_mean'、'item_id_cnt_month_mean'、'item_cnt_month']的[1]偏置
    cols_to_rename=[
                    'item_category_id_cnt_month_mean',
                    'city_code_cnt_month_mean',
                    'subtype_code_cnt_month_mean',
                    'type_code_cnt_month_mean',
                    ]
    shift_range = [1]
    all_set=lag_feature(all_set,shift_range,cols_to_rename)
    
    all_set=all_set.fillna(0)
    all_set = downcast_dtypes(all_set)
    return all_set

# ************************************ 数据处理 *****************************
def date_del(data_path):
    # 从csv文件中将数据导入到DataFrame当中,并对数据进行清洗和剔除,保留有用的数据
    all_set=data_cleansing_integration()
    # all_set.to_csv('%s/all_set.csv' % data_path,index=False)

    # all_set=pd.read_csv('%s/all_set.csv' % data_path)
    # 对数据集中的数据进行特征工程
    all_set=data_feature_extract(all_set)
    
    shops_pred=list(all_set.loc[all_set.date_block_num==34,'shop_id'].drop_duplicates())
    all_set=all_set.loc[all_set.shop_id.isin(shops_pred)]
    ####################3.模型########################
    all_set=all_set[[
                        'shop_id', 
                        'item_id', 
                        'date_block_num', 
                        'city_code', 
                        'item_category_id',
                        'type_code', 
                        'subtype_code', 
                        'item_cnt_month', 
                        # 'item_price_mean',
                        # 'item_price_mean_month', 
                        # 'item_price_mean_month_pre_1',
                        # 'item_price_mean_month_pre_2', 
                        # 'item_price_mean_month_pre_3',
                        # 'item_price_mean_month_pre_4', 
                        # 'item_price_mean_month_pre_5',
                        # 'item_price_mean_month_pre_6', 
                        # 'item_price_delta_month_pre_1',
                        # 'item_price_delta_month_pre_2', 
                        # 'item_price_delta_month_pre_3',
                        # 'item_price_delta_month_pre_4', 
                        # 'item_price_delta_month_pre_5',
                        # 'item_price_delta_month_pre_6', 
                        'item_price_delta_month',
                        'revenue_float_month_pre_1', 
                        'days_of_month',
                        'month', 
                        'year',
                        'shop_code', 
                        'item_category_code', 
                        'item_code',
                        # 'shop_id_cnt_month_mean', 
                        # 'item_category_id_cnt_month_mean',
                        # 'item_id_cnt_month_mean', 
                        # 'city_code_cnt_month_mean',
                        # 'type_code_cnt_month_mean', 
                        # 'subtype_code_cnt_month_mean',
                        'shop_id_cnt_month_mean_pre_1', 
                        'item_id_cnt_month_mean_pre_1',
                        'item_cnt_month_pre_1', 
                        'shop_id_cnt_month_mean_pre_2',
                        'item_id_cnt_month_mean_pre_2', 
                        'item_cnt_month_pre_2',
                        'shop_id_cnt_month_mean_pre_3', 
                        'item_id_cnt_month_mean_pre_3',
                        'item_cnt_month_pre_3', 
                        'shop_id_cnt_month_mean_pre_6',
                        'item_id_cnt_month_mean_pre_6', 
                        'item_cnt_month_pre_6',
                        'shop_id_cnt_month_mean_pre_12', 
                        'item_id_cnt_month_mean_pre_12',
                        'item_cnt_month_pre_12', 
                        'item_category_id_cnt_month_mean_pre_1',
                        'city_code_cnt_month_mean_pre_1', 
                        'subtype_code_cnt_month_mean_pre_1',
                        'type_code_cnt_month_mean_pre_1',
                        'shop_item_first_sale',
                        'shop_item_last_sale',
                        'item_id_first_sale',
                        'item_id_last_sale'
                    ]]
    
    num_first_level_models=3
    meta_size=21
    slice_start = 0
    
    Target = 'item_cnt_month'
    
    meta_months_data=range(12+meta_size,35)       # 
    mask = all_set[date_block_col].isin(meta_months_data)
    y_all_level2 = all_set[Target][mask].values
    X_all_level2 = np.zeros([y_all_level2.shape[0], num_first_level_models])
    
    pre_cols = [col for col in all_set.columns if '_pre_' in col]
    others_cols=['item_category_id','month', 'year','days_of_month','item_price_delta_month',
                 'shop_item_first_sale','shop_item_last_sale','item_id_first_sale','item_id_last_sale']
    id_code=['shop_code', 'item_category_code', 'item_code','city_code','type_code','subtype_code']
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    for cur_block_num in tqdm(meta_months_data):
        # 3.0 建立训练集和测试集
        mask=all_set['date_block_num'].isin(range(cur_block_num-meta_size,cur_block_num))
        train_set=all_set[mask][id_cols+id_code+pre_cols+others_cols].copy()
        train_value=all_set[mask][Target].copy()
        test_set=all_set[all_set[date_block_col]==cur_block_num][id_cols+id_code+pre_cols+others_cols].copy()
        test_value=all_set[all_set[date_block_col]==cur_block_num][Target].copy()
        
        preds=[]
        # 3.1 lightgbm模型
        import lightgbm as lgb
        lgb_params = {
                      'feature_fraction': 0.9,        # 每次迭代的时候随机选择特征的比例，默认为1，训练前选好
                      'metric': 'rmse',                # root square loss(平方根损失）
                      'nthread':3,                    # LightGBM 的线程数
                      'min_data_in_leaf': 2**2,        # 一个叶子上数据的最小数量. 可以用来处理过拟合
                      'bagging_fraction': 0.75,        # 类似于 feature_fraction, 但是它在训练时选特征
                      'learning_rate': 0.02,        # 学习率
                      'objective': 'rmse',            # regression_l2, L2 loss, alias=regression, mean_squared_error, mse
                      # 'bagging_seed': 2**7,            # bagging 随机数种子
                      'num_leaves': 2**11,            # 一棵树上的叶子数
                      'bagging_freq':1,                # bagging 的频率, 0 意味着禁用 bagging. k意味着每 k次迭代执行bagging
                      'verbose':1                    # verbose: 详细信息模式，0 或者 1 
                      }
        estimator = lgb.train(lgb_params, lgb.Dataset(train_set, label=train_value), 500)
        pred_test = estimator.predict(test_set)
        # pred_train = estimator.predict(train_set)
        
        preds.append(pred_test)
        
        # print('Train RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(train_value, pred_train.clip(0,20)))))
        print('Test RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(test_value.values.clip(0,20), pred_test.clip(0,20)))))
        
        slice_end = slice_start + test_set.shape[0]
        X_all_level2[ slice_start : slice_end , :] = np.c_[preds].transpose()        # transpose用于转置的
        slice_start = slice_end
        
        print("运行时长为：%dmin%ds\t 完成一轮训练"%((time.time()-start_time)/60,(time.time()-start_time)%60))
        
        # plt.figure()
        # plt.plot(pred_test.clip(0,20))
        # plt.title("just pre")
        # plt.figure()
        # plt.plot(test_value.values.clip(0,20))
        # plt.title("just test")
        # plt.figure()
        # plt.plot(pred_test.clip(0,20)-test_value.values.clip(0,20))
        # plt.title("just pred-test")
        
        # lgb.plot_importance(estimator, max_num_features=100)
        # plt.title("Featurertances")
        # plt.show()
        
    submission_path="../Result"
    submission = pd.read_csv('%s/sample_submission.csv' % data_path)
    submission['item_cnt_month'] = pred_test.clip(0,20)
    submission[['ID', 'item_cnt_month']].to_csv('%s/submission_Lightgvm.csv' % (submission_path), index = False)
    
    # 4. Ensembling -------------------------------------------------------------------
    test_nrow = len(preds[0])        # 预测的占的长度
    X_train_level2 = X_all_level2[ : -test_nrow, :]        # 训练集（前面预测的第27月到33月的值）test_nrow=214200
    X_test_level2 = X_all_level2[ -test_nrow: , :]        # 测试集（前面预测的第34月值）
    y_train_level2 = y_all_level2[ : -test_nrow]        # 训练集实际值
    
    # A. Second level learning model via linear regression第二层学习,模型1,使用线性回归模型
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train_level2, y_train_level2)
    test_preds_lr_stacking = lr.predict(X_test_level2)
    train_preds_lr_stacking = lr.predict(X_train_level2)
    print('Train R-squared for %s is %f' %('train_preds_lr_stacking', sqrt(mean_squared_error(y_train_level2, train_preds_lr_stacking))))
    
    submission_path="../Result"
    submission = pd.read_csv('%s/sample_submission.csv' % data_path)
    submission['item_cnt_month'] = test_preds_lr_stacking.clip(0,20)
    submission[['ID', 'item_cnt_month']].to_csv('%s/submission_Lightgvm_use_ens.csv' % (submission_path), index = False)
    
    print("运行时长为：%dmin%ds\t 预测完成"%((time.time()-start_time)/60,(time.time()-start_time)%60))
    
    plt.figure()
    plt.plot(pred_test.clip(0,20))
    plt.title("just pre")
    plt.figure()
    plt.plot(submission['item_cnt_month'])
    plt.title("use_en")
    
    print()


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')
    
    import pandas as pd
    pd.set_option('display.max_rows', 99)        # 在控制台显示dataframe数据最多行数,超过后自动省略
    pd.set_option('display.max_columns', 50)     # 在控制台显示dataframe数据最多列数,超过后自动省略
    
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    month_sala_min=0
    month_sala_max=20
    
    import time
    start_time = time.time()
    
    data_path = '../Data'
    
    id_cols=['shop_id', 'item_id']
    date_block_col='date_block_num'
    date_del(data_path)
    