import numpy as np
import pandas as pd
from textblob import TextBlob

def process(df):
    # This is where you can do all your processing
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)

    return df


# Load the dataset
trainingSet = pd.read_csv("C:/Users/dell/Desktop/midterm-Kotoric1-master/train.csv")

# Process the DataFrame
train_processed = process(trainingSet)
print(train_processed.dtypes)
summary=np.array(train_processed['Summary'])
text=np.array(train_processed['Text'])

emotion1=[]
emotion2=[]

for i in range (len(summary)):
    if type(summary[i])==str:
        emotion1.append(TextBlob(summary[i]).sentiment.polarity)
    else:
        emotion1.append('nan')
        # print(i)
# print(emotion[69352])
print(len(emotion1))
train_processed['sentimental1']=emotion1
# print(train_processed.head())
# print(train_processed.dtypes)

for i in range (len(text)):
    if type(text[i])==str:
        emotion2.append(TextBlob(text[i]).sentiment.polarity)
    else:
        emotion2.append('nan')
        # print(i)
# print(emotion[69352])
print(len(emotion2))
train_processed['sentimental2']=emotion2
emotion3=[]
for i in range(len(emotion1)):
    if emotion1[i]== "nan" :
        emotion3.append(emotion2[i])
    elif emotion2[i]== "nan":
        emotion3.append(emotion1[i])
    else:
        emotion3.append((emotion1[i]+emotion2[i])/2)
    
train_processed['sentimental3']=emotion3


# import category_encoders as ce
# encoder1=ce.TargetEncoder(cols='ProductId')
# train_processed['product']=encoder1.fit_transform(train_processed['ProductId'],train_processed['Score'])
# encoder2=ce.TargetEncoder(cols='UserId')
# train_processed['user']=encoder2.fit_transform(train_processed['UserId'],train_processed['Score'])
# print(trainingSet.head())
# # print(trainingSet.dtypes)
# print("train.csv shape is ", train_processed.shape)
# train_processed=train_processed.dropna(axis=0,how='any')  
# Load test set

# from sklearn import preprocessing
# scaler = preprocessing.StandardScaler()
# X_train_processed1 = scaler.fit_transform(train_processed[['Helpfulness','sentimental1','sentimental2','product','user']])
# # print(X_train_processed1[0])
# # print(X_train_processed1[1])
# # print(X_train_processed1[2])
# train_processed['Helpfulness']=X_train_processed1[:,0]
# train_processed['sentimental1']=X_train_processed1[:,1]
# train_processed['sentimental2']=X_train_processed1[:,2]
# train_processed['product']=X_train_processed1[:,3]
# train_processed['user']=X_train_processed1[:,4]
# print(train_processed.head())

submissionSet = pd.read_csv("C:/Users/dell/Desktop/midterm-Kotoric1-master/test.csv")

# Merge on Id so that the test set can have feature columns as well
testX= pd.merge(train_processed, submissionSet, left_on='Id', right_on='Id')
testX = testX.drop(columns=['Score_x'])
testX = testX.rename(columns={'Score_y': 'Score'})

# The training set is where the score is not null
trainX =  train_processed[train_processed['Score'].notnull()]
print("train.csv shape is ", train_processed.shape)
print("test.csv shape is ", testX.shape)
print(train_processed.head())
print(testX.head())
# summary=np.array(testX['Summary'])
# text=np.array(testX['Text'])

# emotion1=[]
# emotion2=[]

# for i in range (len(summary)):
#     if type(summary[i])==str:
#         emotion1.append(TextBlob(summary[i]).sentiment.polarity)
#     else:
#         emotion1.append('nan')
#         # print(i)
# # print(emotion[69352])
# # print(len(emotion))
# testX['sentimental1']=emotion1
# # print(trainingSet.head())
# # print(trainingSet.dtypes)

# for i in range (len(text)):
#     if type(text[i])==str:
#         emotion2.append(TextBlob(text[i]).sentiment.polarity)
#     else:
#         emotion2.append('nan')
#         # print(i)
# # print(emotion[69352])
# # print(len(emotion))
# testX['sentimental2']=emotion2
# # print(trainingSet.head())
# # print(trainingSet.dtypes)
# testX=testX.dropna(axis=0,how='any')  


print(trainX[trainX.isnull().T.any()])
print(testX[testX.isnull().T.any()])

testX.to_csv("./data/X_test.csv", index=False)
trainX.to_csv("./data/X_train.csv", index=False)