import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import scipy.stats as st


pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

de_data_tr = pd.read_csv('./data/happiness_train_complete.csv',encoding='gbk')
de_data_te = pd.read_csv('./data/happiness_test_complete.csv',encoding='gbk')
data_tr = pd.read_csv('./data/happiness_train_abbr.csv')
data_te = pd.read_csv('./data/happiness_test_abbr.csv')
print(de_data_te.head())
print(data_tr.head())


drop_features =['survey_time','work_status','work_yr','work_type','work_manage']
pls_features =['neighbor_familiarity','inc_exp','f_edu','f_work_14','m_edu','m_work_14',
               'son','daughter','s_edu','s_income','class_10_before','class_10_after','class_14']

data = pd.concat([data_tr,data_te],ignore_index=True,sort=False)
data_cp = data
data = data.drop(drop_features,axis =1)
print(data.head())
print(data.shape)



#拼接新的列
de_data = pd.concat([de_data_te,de_data_tr],ignore_index=True,sort=False)
pls_data = de_data[pls_features]
data = pd.concat([data,pls_data],axis=1,sort=False)

#处理异常值
data['family_income'].fillna(data.loc[data['family_income'].isnull(),'income']*2,inplace=True)
for index,row in data.iterrows():
    if row['family_income']<row['income']:
        row['income'] = row['family_income']/2

for col in data.columns:
    data.loc[data[col] < 0, col] = data[col].mode().iloc[0]
    if col != 'happiness':
        data[col].fillna(data[col].mode().iloc[0],inplace = True)

#构造一个阶级的特征
for index, row in data.iterrows():
    if row['class'] < row['class_10_before'] or row['class'] < row['class_14']:
        data['class_down'] = 1
    else:
        data['class_down'] = 0
    if row['class'] > row['class_10_before'] or row['class'] > row['class_14']:
        data['class_up'] = 1
    else:
        data['class_up'] = 0
    if row['class_10_after'] > row['class']:
        data['class_exp'] = 1
    else:
        data['class_exp'] = 0


#处理年龄
data['age'] = 2015 - data['birth']
del data['birth']
bin = [ i * 10 for  i in range(11)]
data['age'] = pd.cut(data['age'], bin , labels=False)

#treefeatures
#收入比
data['income/s_income'] = data['income']/(data['s_income']+1)
data['income+s_income'] = data['income']+(data['s_income']+1)
data['income/family_income'] = data['income']/(data['family_income']+1)
data['all_income/family_income'] = (data['income']+data['s_income'])/(data['family_income']+1)
data['income/inc_exp'] = data['income']/(data['inc_exp']+1)
data['family_income/m'] = data['family_income']/(data['family_m']+0.01)

#收入/面积比
data['income/floor_area'] = data['income']/(data['floor_area']+0.01)
data['all_income/floor_area'] = (data['income']+data['s_income'])/(data['floor_area']+0.01)
data['family_income/floor_area'] = data['family_income']/(data['floor_area']+0.01)

data['income/m'] = data['floor_area']/(data['family_m']+0.01)

#class
data['class_10_diff'] = (data['class_10_after'] - data['class_10_before'])
data['class_diff'] = data['class'] - data['class_10_before']
data['class_14_diff'] = data['class'] - data['class_14']


#province mean
data['province_income_mean'] = data.groupby(['province'])['income'].transform('mean').values
data['province_family_income_mean'] = data.groupby(['province'])['family_income'].transform('mean').values
data['province_equity_mean'] = data.groupby(['province'])['equity'].transform('mean').values
data['province_depression_mean'] = data.groupby(['province'])['depression'].transform('mean').values
data['province_floor_area_mean'] = data.groupby(['province'])['floor_area'].transform('mean').values

#city   mean
data['city_income_mean'] = data.groupby(['city'])['income'].transform('mean').values
data['city_family_income_mean'] = data.groupby(['city'])['family_income'].transform('mean').values
data['city_equity_mean'] = data.groupby(['city'])['equity'].transform('mean').values
data['city_depression_mean'] = data.groupby(['city'])['depression'].transform('mean').values
data['city_floor_area_mean'] = data.groupby(['city'])['floor_area'].transform('mean').values


#county  mean
data['county_income_mean'] = data.groupby(['county'])['income'].transform('mean').values
data['county_family_income_mean'] = data.groupby(['county'])['family_income'].transform('mean').values
data['county_equity_mean'] = data.groupby(['county'])['equity'].transform('mean').values
data['county_depression_mean'] = data.groupby(['county'])['depression'].transform('mean').values
data['county_floor_area_mean'] = data.groupby(['county'])['floor_area'].transform('mean').values

#ratio 相比同个城市
data['income/county'] = data['income']/(data['county_income_mean']+1)
data['family_income/county'] = data['family_income']/(data['county_family_income_mean']+1)
data['equity/county'] = data['equity']/(data['county_equity_mean']+1)
data['depression/county'] = data['depression']/(data['county_depression_mean']+1)
data['floor_area/county'] = data['floor_area']/(data['county_floor_area_mean']+1)

#age   mean
data['age_income_mean'] = data.groupby(['age'])['income'].transform('mean').values
data['age_family_income_mean'] = data.groupby(['age'])['family_income'].transform('mean').values
data['age_equity_mean'] = data.groupby(['age'])['equity'].transform('mean').values
data['age_depression_mean'] = data.groupby(['age'])['depression'].transform('mean').values
data['age_floor_area_mean'] = data.groupby(['age'])['floor_area'].transform('mean').values
data['age_health_mean'] = data.groupby(['age','gender'])['health'].transform('mean').values
data['age_edu_mean'] = data.groupby(['age','gender'])['edu'].transform('mean').values


#age/gender   mean
data['age_income_mean'] = data.groupby(['age','gender'])['income'].transform('mean').values
data['age_family_income_mean'] = data.groupby(['age','gender'])['family_income'].transform('mean').values
data['age_equity_mean'] = data.groupby(['age','gender'])['equity'].transform('mean').values
data['age_depression_mean'] = data.groupby(['age','gender'])['depression'].transform('mean').values
data['age_floor_area_mean'] = data.groupby(['age','gender'])['floor_area'].transform('mean').values
data['age_gender_health_mean'] = data.groupby(['age','gender'])['health'].transform('mean').values


#class   mean
data['city_class_income_mean'] = data.groupby(['class'])['income'].transform('mean').values
data['city_class_family_income_mean'] = data.groupby(['class'])['family_income'].transform('mean').values
data['city_class_equity_mean'] = data.groupby(['class'])['equity'].transform('mean').values
data['city_class_depression_mean'] = data.groupby(['class'])['depression'].transform('mean').values
data['city_class_floor_area_mean'] = data.groupby(['class'])['floor_area'].transform('mean').values



print(data['income'].value_counts())

nn_data_train = data[:8000]
nn_data_test = data[8000:]
nn_data_train.to_csv('data/tree_data_train.csv',index = False,sep = ' ')
nn_data_test.to_csv('data/tree_data_test.csv',index = False, sep = ' ')








