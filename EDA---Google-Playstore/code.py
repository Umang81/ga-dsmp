# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
data['Rating'].hist()
data = data[data['Rating']<=5]
data['Rating'].hist()
#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = total_null/data.isnull().count()
missing_data = pd.concat([total_null,percent_null], axis=1, keys=['Total','Percent'])
print(missing_data)

  
# making new data frame with dropped NA values 
data.dropna(axis = 0, how ='any',inplace=True) 
total_null_1 = data.isnull().sum()
percent_null_1 = total_null_1/data.isnull().count()
missing_data_1 = pd.concat([total_null_1,percent_null_1], axis=1, keys=['Total','Percent'])
print(missing_data_1)

# code ends here


# --------------

#Code starts here
g=sns.catplot(x='Category',y='Rating',data=data,kind = 'box',height=10)
g.set_xticklabels(rotation=90)
g.set_titles('Rating vs Category [BoxPlot]')

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here



data['Installs'] = data['Installs'].str.replace(',', '')
data['Installs'] = data['Installs'].str.replace('+', '')
data['Installs'] = data['Installs'].astype(int)
print(data['Installs'].value_counts())

le= LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])

g = sns.regplot(x='Installs',y='Rating',data=data)
g.set_title('Rating vs Installs [Regplot]')
#Code ends here



# --------------
#Code starts here

data['Price']=data['Price'].str.replace('$','').astype(float)
print(data['Price'].value_counts())
g = sns.regplot(x='Price',y='Rating',data=data)
g.set_title('Rating vs Price [RegPlot]')

#Code ends here


# --------------

#Code starts here
data['Genres']=data['Genres'].str.split(';').str[0]

gr_mean = data[['Genres','Rating']].groupby('Genres',as_index=False).mean()
gr_mean.sort_values('Rating',inplace = True)
print(gr_mean.iloc[0,:])
print(gr_mean.iloc[-1,:])

#Code ends hereby 


# --------------

#Code starts here
print(data["Last Updated"])
data['Last Updated']=pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()
print(max_date)
data['Last Updated Days'] = (max_date-data['Last Updated']).dt.days
print(data['Last Updated Days'])
g = sns.regplot(x='Last Updated Days',y='Rating',data=data,color='teal')
g.set_title('Rating vs Last Updated [RegPlot]')
#Code ends here


