# Problem statement

# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. 
# Also, certain attributes of each product and store have been defined. 
# The aim is to build a predictive model and find out the sales of each product at a particular store.
# Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.


#Hypothesis generation

#Store level hypothesis

#City type: Stores located in urban or Tier 1 cities should have higher sales because of the higher income levels of people there.
#Population density: Stores located in densly populated areas will have higher sales because of higher demand
#Store capacity:  Stores with bigger sizes will generate more sales because consumers will be more likely to buy all items there rather than using different stores
#Competitors: STores located near competitors will have fewer sales because of more competition
#Marketing: Stores with effective marketing will have higher sales due to their ability to attract customers
#Ambience: Stores that are well maintained with good staff will sell higher due to increased reputation

#Product level hypothesis
#BVarded products will sell more due to brand loyalty and indication of quality
#Product placement: Products optimally placed will have higher sales
#Essential vs non essential: Products which are essential (Toilet roll, washing poweder) will sell mopre consistentyly than impulse purchases
#Promotionasl opffers: If a product is on sale it will sell more

#Import libraries

#Data processing
import pandas as pd
import numpy as numpy

#Graphics
import seaborn as sns
import matplotlib.pyplot as pyplot

# Import data sets

train = pd.read_csv('/Users/marc/Data-science-portfolio/Bigmart sales data/dataset/Train_UWu5bXk.csv')
test = pd.read_csv('/Users/marc/Data-science-portfolio/Bigmart sales data/dataset/Test_u94Q5KV.csv')

train['source']='train'
test['source']='test'

data = pd.concat([train, test], ignore_index=True)

# Data exploration

data.columns

data.dtypes            #11 independent variables and 1 dependent variable which is item outlet sales
data.describe()

#Some observations
#Item visibility has a mininum value of zero, this could mean the product is not on display and is just in storage, however all products are selling some so this does not seem to completely make sense
#It may be better to convert establishment year to years open (Establioshment year-current year)

data['Item_Weight'].describe() #Mean should be adequate to subsitute in for item weight
data['Outlet_Size'].describe() #Majority of the stores seem to be medium so this will be used for null values here

# There are 7 categorical variables

data.apply(lambda x: len(x.unique()))

#There are 1559 unique products and 10 stores. The store only has 16 different types of products

#Filter categorical variables
categorical_columns = [ x for x in data.dtypes.index if data.dtypes[x]=='object']

#Drop identifiers since they won't be useful to this analysis
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier', 'Outlet_Identifier']]

for column in categorical_columns:
    print('Frequency of variable %s' %column)
    print(data[column].value_counts())

#Some low fat goods are mis coded as LF
#Not all categoreies for item type have substantial num,bers and cvould be combined with others for a better insight


## Data cleaning
#Removing empty values 

data.isnull().sum()    #There are a number of null values in item weight and outlet size which must be fixed

#Replace item weight with the avergae weight of the item
item_average_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier') #Calculate average weight for each item type
miss_bool = data['Item_Weight'].isnull() #Get index of missing weights
# data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight[x])
data.loc[miss_bool,'Item_Weight'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_average_weight.at[x,'Item_Weight'])
data['Item_Weight'].isnull().sum()

#Do the same with outlet size and the mode
from scipy.stats import mode
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: mode(x).mode[0])) #Calculate the mode for each store type
miss_bool = data['Outlet_Size'].isnull() #Get index of missing weights
data.loc[miss_bool,'Outlet_Size'] = data.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
data['Outlet_Size'].isnull().sum()

## Feature engineering

# Consider combining store types if their average sales are similar
data.pivot_table(values='Item_Outlet_Sales', index='Outlet_Type')
#The vareince is reasonably high so we'll leave it as is

#Item visability is zero for some items which doesn't make much sense so this must be changed
visibility_average = data.pivot_table(values='Item_Visibility', index='Item_Identifier')

#Impute 0 values with the meanvisibility of that product
miss_bool = data['Item_Visibility']==0 #Get index of missing weights
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_average.at[x,'Item_Visibility'])

# Create a new variable of visibility score which is visibility divided by average of that type
data['vis_mean_ratio'] = data.apply(lambda x: x['Item_Visibility']/visibility_average.loc[x['Item_Identifier']], axis=1)

data['vis_mean_ratio'].describe()

#Recategorise the food types
data['item_id_type'] = data['Item_Identifier'].apply(lambda x: x[0:2]) #Get first two characters of id
data['item_id_type'] = data['item_id_type'].map({'FD': 'Food', 'DR': 'Drink', 'NC': 'Non_Consumable'})
data['item_id_type'].value_counts()

#Change store opening to outlet years
data['years_open'] = 2013 - data['Outlet_Establishment_Year']

# Some fat values were mislabled so this should be sorted

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

#Some Non consumables have a fat value as well so this should be removed
data.loc[data['item_id_type']=='Non_Consumable', 'Item_Fat_Content'] = "Non-edible"

data['Item_Fat_Content'].value_counts()

#Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
le = LabelEncoder()

cat_vars = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'item_id_type', 'Outlet']
for i in cat_vars:
    data[i]=le.fit_transform(data[i])

pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'item_id_type', 'Outlet'])
data.dtypes

# Drop values which are no longer useful
data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

#Split back to train and tes
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']

train.drop(['source'], axis=1, inplace=True)
test.drop(['source', 'Item_Outlet_Sales'], axis=1, inplace=True)

# Model Building