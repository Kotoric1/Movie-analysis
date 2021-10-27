from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import seaborn as sns
from sklearn import preprocessing
import math

trainingSet = pd.read_csv("./data/X_train.csv")

# train=trainingSet.dropna(axis=0,how='any')                  #delete the rows with nan value   
# summary=np.array(trainingSet['Summary'])
# text=np.array(trainingSet['Text'])

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
# trainingSet['sentimental1']=emotion1
# print(trainingSet.head())
# print(trainingSet.dtypes)

# for i in range (len(text)):
#     if type(text[i])==str:
#         emotion2.append(TextBlob(text[i]).sentiment.polarity)
#     else:
#         emotion2.append('nan')
#         # print(i)
# # print(emotion[69352])
# # print(len(emotion))
# trainingSet['sentimental2']=emotion2
# print(trainingSet.head())
print(trainingSet.dtypes)


train=trainingSet.dropna(axis=0,how='any')  

print(train.head())
# helpfulness=np.array(train['Helpfulness']) 
# print(np.unique(helpfulness))
# X=train.drop(['Score'],axis=1)
# y=train['Score']

X_train, X_test, Y_train, Y_test = train_test_split(
        train.drop(['Score'], axis=1),
        train['Score'],
        test_size=1/4.0,
        random_state=0
    )

X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary'])
X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary'])

# # GBDT
# from sklearn.ensemble import GradientBoostingClassifier

# model = GradientBoostingClassifier()
# gbdt=model.fit(X_train_processed, Y_train)
# # print(gbdt)

# gbdt_pred = gbdt.predict(X_test_processed)
# # print(gbdt_pred)

# print("RMSE on testing set = ", mean_squared_error(Y_test, gbdt_pred))
# #1.37

#Kmeans
# SSE = []  
# for k in range(1,9):                            #plot the graph to see which k valuse I should choose
#     estimator = KMeans(n_clusters=k)            #using elbow method to determine the k value
#     estimator.fit(X_train_processed)
#     SSE.append(estimator.inertia_) 
# X = range(1,9)
# plt.xlabel('k')
# plt.ylabel('SSE')
# plt.plot(X,SSE,'o-')
# plt.show()                  #found cluster number=3 is best

# kmeans = KMeans(n_clusters=4, random_state=0).fit(X_train_processed, Y_train)
# kmeans_pred =kmeans.predict(X_test_processed)

# print("RMSE on testing set = ", mean_squared_error(Y_test, kmeans_pred))


# kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train_processed, Y_train)
# kmeans_pred =kmeans.predict(X_test_processed)

# print("RMSE on testing set = ", mean_squared_error(Y_test, kmeans_pred))
#RMSE=11.16 too large

# KNN
# model = KNeighborsClassifier(n_neighbors=3).fit(X_train_processed, Y_train)

# # # Predict the score using the model
# KNN_predictions = model.predict(X_test_processed)

# # # Evaluate your model on the testing set
# print("RMSE on testing set = ", mean_squared_error(Y_test, KNN_predictions))
# 1.26



# linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model=model.fit(X_train_processed, Y_train)
linreg_pred=model.predict(X_test_processed)
print("RMSE on testing set = ", mean_squared_error(Y_test, linreg_pred))
#0.73

# print(max(list(linreg_pred)))
# print(min(list(linreg_pred)))
# print(np.sum(linreg_pred)/len(linreg_pred))
# df = pd.DataFrame(columns = ["test"])
# df['test']=linreg_pred
# # print(df.describe())
# df['test'].value_counts().plot(kind='bar', legend=True, alpha=.5)
# plt.show()

# print(linreg.coef_)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# param_grid = [{'penalty':['l1', 'l2', 'elasticnet', 'none'],'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],'C':[0.001,0.01,0.1,1]}]

# logreg=LogisticRegression()
# grid_search = GridSearchCV(logreg, param_grid, cv=3,
#                           scoring='neg_mean_squared_error')
# grid_search.fit(X_train_processed, Y_train)
# print(grid_search.best_params_)
# print(grid_search.best_estimator_)
# import sklearn.preprocessing
# scaler = preprocessing.StandardScaler()
# X_train_processed1 = scaler.fit_transform(X_train_processed)
# # print(X_train_processed1[0])
# # print(X_train_processed1[1])
# # print(X_train_processed1[2])
# X_train_processed['Helpfulness']=X_train_processed1[:,0]
# X_train_processed['sentimental1']=X_train_processed1[:,1]
# X_train_processed['sentimental2']=X_train_processed1[:,2]
# X_train_processed['product']=X_train_processed1[:,3]
# X_train_processed['user']=X_train_processed1[:,4]
# print(X_train_processed.head())

# model = LogisticRegression(C= 1, penalty= 'l2', solver= 'lbfgs',random_state=0).fit(X_train_processed, Y_train)
# logreg_pred=model.predict(X_test_processed)
# print("RMSE on testing set = ", mean_squared_error(Y_test, logreg_pred))

# fig, ax = plt.subplots()
# ax.scatter(Y_test, linreg_pred)
# ax.plot([np.min(Y_test), np.max(Y_test)], [np.min(linreg_pred), np.max( linreg_pred)], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()

#1.02
# cm = confusion_matrix(Y_test, linreg_pred, normalize='true')
# sns.heatmap(cm, annot=True)
# plt.title('Confusion matrix of the classifier')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()




X_submission = pd.read_csv("./data/X_test.csv")
# print(X_submission.columns)
# print(X_submission.head())

#,'HelpfulnessNumerator','HelpfulnessDenominator','Time','sentimental3'
X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'Score'])


sentimental1=np.array(X_submission_processed['sentimental1'])
sentimental2=np.array(X_submission_processed['sentimental2'])
sentimental1[2001]=sentimental2[2001]
sentimental2[18795]=sentimental1[18795]
sentimental1[41742]=sentimental2[41742]
sentimental2[43189]=sentimental1[43189]
sentimental1[113410]=sentimental2[113410]
sentimental2[153210]=sentimental1[153210]
sentimental2[180980]=sentimental1[180980]
sentimental2[191684]=sentimental1[191684]
sentimental2[197635]=sentimental1[197635]
sentimental2[220075]=sentimental1[220075]
sentimental2[221862]=sentimental1[221862]
sentimental1[242129]=sentimental2[242129]
sentimental1[258990]=sentimental2[258990]
sentimental1[264697]=sentimental2[264697]
sentimental2[281151]=sentimental1[281151]


X_submission_processed['sentimental1']=sentimental1
X_submission_processed['sentimental2']=sentimental2
# X_submission_processed['sentimental3']=sentimental3



# product=np.array(X_submission_processed['product'])
# user=np.array(X_submission_processed['product'])
# print('mean:',train['Score'].mean())
# for i in [5801,7478,8241,15078,70424,94653,190601,267639]:
#     if(product[i]=='nan'):
#          product[i]=user[i]
#     else:
#         user[i]=product[i]
# for i in[127393,127394,127395,127396,127397,208835,208836,208837,208838,208839,233194,239389]:
#         product[i]=train['Score'].mean()
#         user[i]=train['Score'].mean()
# X_submission_processed['product']=product
# X_submission_processed['user']=user
# error=X_submission_processed[X_submission_processed.isnull().T.any()]
# print(error)

X_submission['Score'] = model.predict(X_submission_processed)

# Create the submission file
submission = X_submission[['Id', 'Score']]

score=list(submission ['Score'])
print(max(score))
print(min(score))
outlier=[]
for i in range(len(score)):
    if(score[i]>5.0 or score[i]<1.0): 
        outlier.append(i)

print(len(outlier))

for i in range(len(outlier)):
    if(score[outlier[i]]>5.0):
        score[outlier[i]]=5.0
    else:
        score[outlier[i]]=1.0
# for i in range(len(score)):
#     score[i]=math.ceil(score[i])
submission ['Score']=score
print(submission.describe())
submission.to_csv("submission.csv", index=False)



