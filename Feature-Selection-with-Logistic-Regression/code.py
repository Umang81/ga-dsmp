# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset
dataset = pd.read_csv(path)

# look at the first five columns
print(dataset.head())

# Check if there's any column which is not useful and remove it like the column id
dataset = dataset.drop('Id',axis=1)

# check the statistical description
print(dataset.describe())


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns

#number of attributes (exclude target)
size = len(cols)-1

#x-axis has target attribute to distinguish between classes
x = cols[size]

#y-axis shows values of an attribute
y = cols[0:size]

#Plot violin for all attributes
for i in range (size):
    sns.violinplot(data=dataset,x=x,y=y[i])
    plt.show()


# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
subset_train = dataset.iloc[:,0:10]
data_corr = subset_train.corr()

data_corr = subset_train.corr()
fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(data_corr,cmap='PiYG',annot=True,fmt=".2f",linewidths=.5,ax=ax)

correlation = data_corr.unstack().sort_values(kind='quicksort')
corr_var_list = correlation[((correlation>upper_threshold)|(correlation<lower_threshold))&(correlation != 1)]
# Code ends here




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)



# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
X = dataset.drop('Cover_Type',1)
y = dataset['Cover_Type']
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)

#Standardized
#Apply transform only for non-categorical data
scaler = StandardScaler()
X_train_temp = scaler.fit_transform(X_train)
X_test_temp = scaler.transform(X_test)

#Concatenate non-categorical data and categorical
scaler = StandardScaler()
X_train_temp = scaler.fit_transform(X_train.iloc[:,0:10])
X_test_temp = scaler.fit_transform(X_test.iloc[:,0:10])

X_train1=numpy.concatenate((X_train_temp,X_train.iloc[:,10:]),axis=1)
X_test1=numpy.concatenate((X_test_temp,X_test.iloc[:,10:]),axis=1)

scaled_features_train_df = pd.DataFrame(X_train1,columns=X_train.columns,index=X_train.index)
scaled_features_test_df = pd.DataFrame(X_test1,columns=X_test.columns,index=X_test.index)



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
skb = SelectPercentile(score_func = f_classif,percentile =20)
predictors = skb.fit_transform(X_train1,Y_train)

scores = skb.scores_
Features = scaled_features_train_df.columns

dataframe1 = pd.DataFrame(Features)
dataframe2 = pd.DataFrame(scores)
dataframe = pd.concat((dataframe1,dataframe2),axis=1)
dataframe.columns = ['Features','scores']
dataframe.sort_values('scores',ascending=False,inplace=True)

top_k_predictors = list(dataframe[dataframe.scores>dataframe.scores.quantile(0.8)]['Features'])
print(top_k_predictors)


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

model = LogisticRegression()
clf = clf1 = OneVsRestClassifier(estimator = model)

model_fit_all_features = clf1.fit(X_train,Y_train)
predictions_all_features = clf1.predict(X_test)
score_all_features = accuracy_score(Y_test,predictions_all_features)

model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors],Y_train)
predictions_top_features = clf.predict(scaled_features_test_df[top_k_predictors])
score_top_features = accuracy_score(Y_test,predictions_top_features)


