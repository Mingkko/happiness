import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import scipy.stats as st
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

#先简易观察数据
path = 'data/'
train_data = pd.read_csv(path+'happiness_train_abbr.csv')
test_data = pd.read_csv(path+'happiness_test_abbr.csv')
print(train_data.head(),train_data.info(),train_data.describe(),train_data.shape)
print(test_data.head(),test_data.info(),test_data.describe(),test_data.shape)

#缺失值分析
missing = train_data.isnull().sum()
missing.sort_values(inplace=True)
missing.plot.bar()
msno.matrix(train_data,figsize=(14,8))


#直接删除缺失列
train_data = train_data.drop(['work_status','work_yr','work_type','work_manage'],axis=1)
#处理一下异常值 负数用众数填充
del train_data['survey_time']
for col in train_data.columns:
    train_data.loc[train_data[col] <= 0, col] = train_data[col].mode().iloc[0]
train_data['family_income'].fillna(train_data.loc[train_data['family_income'].isnull(),'income']*2,inplace=True)

#处理一下年龄，把出生日期变成年龄
train_data['age'] = 2020 - train_data['birth']
del train_data['birth']
#把年龄进行分箱
bin = [i*10 for i in range(11)]
train_data['age'] = pd.cut(train_data['age'],bin,labels=False)

#看看收入的情况
print(train_data['income'].value_counts())
#有几个900w以上的，但是让我奇怪的是，填的有点太精确了，比如9991500，903w，994w这种，暂且认为这些人收入千万吧,看一下他们的信息
print(train_data[train_data['income']>=9000000])
#第一位是20多岁的男性，收入千万，认为自己比同龄人的社会经济地位低，有时会沮丧抑郁，说不上幸福或不幸福，很少休息放松
#关键是，这几个人全家家庭总收入居然低于自己的收入，这是为什么？
#我觉得收入可能对幸福感的影响很大，在特征工程时，应该对这类异常数据好好处理
#处理一下income列
train_data['income'] = np.log1p(train_data['income'])

#挑几列，看看数据的分布
fig,ax = plt.subplots(3,2)
fig.set_size_inches(12,20)
sns.countplot(train_data['edu'],palette ='Spectral_r' ,ax = ax[0][0]).set_title('edu')
sns.distplot(train_data['income'],hist=True ,kde=False,rug=False,ax=ax[0][1]).set_title('income')
sns.countplot(train_data['age'],palette='Spectral_r',ax=ax[1][0]).set_title('age')
sns.countplot(train_data['health'],ax = ax[1][1]).set_title('health')
sns.countplot(data=train_data,x=train_data['depression'],hue="gender",ax = ax[2][0]).set_title('depression')
sns.countplot(train_data['work_exper'],palette='Paired_r',ax =ax[2][1]).set_title('work')
#挑了几列感觉比较重要的列画了图，从图上可以看出受访者大部分为初中学历，年龄在40-60之间，中年人，社会的中流砥柱。
#在抑郁和沮丧方面大部分表现还可以，虽然女性高一些但这是因为样本中女性多。
#健康方面大部分人比较健康，工作岗位方面一言难尽，因为work_status已经删除，不能看出具体岗位，从图上可以看到很多人没有工作，但从年龄上来看这是因为有人已经退休了，非农比务农要高一点
#最后说一下收入，


#继续画图，看一下城市和农村情况，人们对于社会公平的看法，省份分布和等级分布
fig,ax = plt.subplots(1,2)
fig.set_size_inches(12,14)
sns.countplot(train_data['equity'],palette='Spectral_r', ax=ax[0]).set_title('equity')
sns.countplot(train_data['survey_type'],palette='Blues',ax=ax[1]).set_title('survey_type')
lis = train_data['province'].value_counts()
labels = lis.index
plt.rcParams['font.sans-serif']='SimHei'
plt.figure(figsize=(15,15))
plt.title('各省份占比' , fontdict={'fontsize':18})
plt.pie(lis,labels=labels,autopct='%.2f%%',startangle=90,counterclock=False,
        colors=sns.color_palette('RdBu',n_colors=20))
lis2 = train_data['class'].value_counts()
labels2 = lis2.index
plt.figure(figsize=(15,15))
plt.title('等级分布')
plt.pie(lis2,labels=labels2,autopct='%.2f%%',startangle=90,counterclock=False,
        colors=sns.color_palette('RdBu',n_colors=20))
#从图上可以看出，受访者中来自城市偏多一些，大部分人认为社会是公平的，但认为不公平和持中立态度的人数量也很多
#从省份分布上来看来自湖北省的人较多
#等级的图可以看出大部分人认为自己属于中间位置，但345占了超过一半，说明整体中等偏下。
#等级应该是个可以挖掘的点，应该利用详细信息里面的数据继续画图观察10年前，和10年后的情况
#特征工程时也可以做一个与之相关的特征，例如10年后的预期是否比当前高，这代表对未来的期望，还有现在的等级是否比10年前高，这代表现在是否能获取一些满足感和成就感
#还有一项是14岁时家庭等级，也可以用来做比较，代表是否遭遇阶级滑落或者飞越
#这里略过了
# lis3 = train_data['class_10_before'].value_counts()
# labels3 = lis3.index
# plt.figure(figsize=(15,15))
# plt.title('等级分布')
# plt.pie(lis3,labels=labels3,autopct='%.2f%%',startangle=90,counterclock=False,
#         colors=sns.color_palette('BuGn',n_colors=20))
# lis4 = train_data['class_10_after'].value_counts()
# labels4 = lis4.index
# plt.figure(figsize=(15,15))
# plt.title('等级分布')
# plt.pie(lis4,labels=labels4,autopct='%.2f%%',startangle=90,counterclock=False,
#         colors=sns.color_palette('GnBu',n_colors=20))
plt.show()