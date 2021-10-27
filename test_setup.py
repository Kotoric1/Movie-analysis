import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.algorithms import unique
import math


trainingSet = pd.read_csv("C:/Users/dell/Desktop/midterm-Kotoric1-master/train.csv")
testingSet = pd.read_csv("C:/Users/dell/Desktop/midterm-Kotoric1-master/test.csv")
train=trainingSet.dropna(axis=0,how='any')

print("train.csv shape is ", trainingSet.shape)
print("new_train.csv shape is ", train.shape)
print("test.csv shape is ", testingSet.shape)

print()
# score=list(trainingSet['Score'])
# print(max(score))
print(trainingSet.head())
# print()
print(trainingSet.dtypes)

# print()

print(trainingSet.describe())

trainingSet['Score'].value_counts().plot(kind='bar', legend=True, alpha=.5)
plt.show()

# print()
# print("EVERYTHING IS PROPERLY SET UP! YOU ARE READY TO START")

# for i in trainingSet.dtypes.index:
#     print (i)

# print(trainingSet['UserId'])
# print(trainingSet['UserId'].count)
# id=np.array(trainingSet['UserId'])
# product_id=np.array(trainingSet['ProductId'])
# unique_product_id=np.unique(product_id)
# unique_id=np.unique(id)
# print("number of unique ids:",len(unique_id))
# print("number of unique product ids :",len(unique_product_id))
# print(np.array(trainingSet['Score']))
# print(np.unique(np.array(trainingSet['Score']))[4].dtype)
# movie_5=[]
# score=np.array(trainingSet['Score'])


# import pymongo
# client = pymongo.MongoClient('localhost',27017)
# db = client['midterm']
# df = pd.DataFrame(data=trainingSet)
# data = df.to_dict('records')
# db.train.insert_many(data)

# for i in range (len(unique_product_id)):
#     count_5=db.train.count_documents({'Score':5.,'ProductId': unique_product_id[i]})
#     movie_5.append(count_5)
# most_famous_movie=unique_id[movie_5.index(max(movie_5))]
# print(most_famous_movie)

# a=np.unique(np.array(trainingSet[trainingSet['Score']=="5."]))

# print("most popular product id is",trainingSet[trainingSet['Score']==5.]['ProductId'].mode())
# print(trainingSet[trainingSet['ProductId']=='B001KVZ6HK'])


# table=pd.pivot_table(trainingSet,index=['ProductId'],values=['Score'],aggfunc={'Score':np.mean},fill_value=0)

# print(table.columns)
# # print(table.max())
# print(table.sort_values(by=['Score'],ascending=False))
# # print(table.max())
# print(table[table['Score']==5.0])

# train=trainingSet.dropna(axis=0,how='any')  
# product_id=np.array(trainingSet['ProductId'])
# unique_product_id=np.unique(product_id)
# average=[]
# for i in range(len(unique_product_id)):
#     a=train[train['ProductId']==unique_product_id[i]].mean()
#     average.append(a)
    
# print(average[0])
# print(average[1])
# print(average[2])
# print(average[3])
# print(average[4])
# print(len(average))
# print(len(unique_product_id))
