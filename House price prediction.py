#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction Model

# In[53]:


import pandas as pd
import numpy as np


# In[54]:


data = pd.read_csv('Bengaluru_House_Data.csv')


# In[55]:


data.head()


# In[56]:


data.shape


# In[57]:


data.info()


# In[58]:


for column in data.columns :
    print(data[column].value_counts())
    print("-"* 20)


# In[59]:


data.isna().sum()


# In[60]:


data.drop(columns = ['area_type', 'availability', 'society', 'balcony'], inplace= True)


# In[61]:


data.describe()


# In[62]:


data.info()


# In[63]:


data['location'].value_counts()


# In[64]:


data['location'] = data['location'].fillna('Whitefield')


# In[65]:


data['size'].value_counts()


# In[66]:


data['size'] = data['size'].fillna('2 BHK')


# In[68]:


data['bath'] = data['bath'].fillna(data['bath'].median())


# In[69]:


data.info()


# In[70]:


data['bhk'] = data['size'].str.split().str.get(0).astype(int)


# In[71]:


data.head()


# In[72]:


data[data.bhk > 20]


# In[73]:


data['total_sqft'].unique()


# In[74]:


def convert_range(x):
    temp = x.split('-')
    if(x == 2):
        return (float(temp[0])+ float(temp[1]))/2
    try:
        return float(x)
    except:
        return None


# In[75]:


data['total_sqft'] = data['total_sqft'].apply(convert_range)


# In[76]:


data.head()


# In[77]:


data.drop(columns = 'size')


# # Price per square feet column

# In[78]:


data['price_per_sqft'] = data['price']*100000/ data['total_sqft']


# In[79]:


data.head()


# In[80]:


data.describe()


# In[81]:


data['location'].value_counts()


# In[82]:


data['location'] = data['location'].apply(lambda x: x.strip())
location_count = data['location'].value_counts()


# In[83]:


location_count


# In[84]:


location_count_less_10 = location_count[location_count <= 10]
location_count_less_10


# In[85]:


data['location'] = data['location'].apply(lambda x:'other' if x in location_count_less_10 else x)


# In[86]:


location_count


# In[87]:


data['location'].head()


# In[88]:


data['location'].value_counts()


# ## Outlier detection and removal

# In[89]:


data.describe()


# In[90]:


(data['total_sqft']/data['bhk']).describe()


# In[91]:


data = data[(data['total_sqft']/data['bhk']) >= 300]
data.describe()


# In[92]:


data.shape


# In[93]:


data['price_per_sqft'].describe()


# In[94]:


def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        
        st = np.std(subdf.price_per_sqft)
        
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output, gen_df], ignore_index = True)
    return df_output
data = remove_outliers_sqft(data)
data.describe()


# In[95]:


data.shape


# In[97]:


def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')


# In[98]:


data = bhk_outlier_remover(data)


# In[99]:


data.shape


# In[100]:


data


# In[101]:


data.drop(columns=['size','price_per_sqft'], inplace = True)


# ## Cleaned Data

# In[102]:


data.head()


# In[103]:


data.to_csv("Cleaned_data.csv")


# In[104]:


X = data.drop(columns =['price'])
Y = data['price']


# In[105]:


from sklearn.model_selection import train_test_split


# In[107]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[109]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state =0)


# In[110]:


print(X_train.shape)
print(X_test.shape)


# ## Applying Linear Regression

# In[117]:


column_trans = make_column_transformer((OneHotEncoder(sparse =False ), ['location']),
                                     remainder = 'passthrough')


# In[118]:


scaler = StandardScaler()


# In[122]:


lr = LinearRegression()


# In[123]:


pipe = make_pipeline(column_trans,scaler, lr)


# In[124]:


pipe.fit(X_train, Y_train)


# In[125]:


Y_pred_lr = pipe.predict(X_test)


# In[128]:


r2_score(Y_test, Y_pred_lr)


# ## Applying Lasso

# In[129]:


lasso = Lasso()


# In[130]:


pipe = make_pipeline(column_trans, scaler, lasso)


# In[131]:


pipe.fit(X_train, Y_train)


# In[132]:


Y_pred_lasso = pipe.predict(X_test)
r2_score(Y_test, Y_pred_lasso)


# ## Applying Ridge

# In[133]:


ridge = Ridge()


# In[134]:


pipe = make_pipeline(column_trans, scaler, ridge)


# In[135]:


pipe.fit(X_train, Y_train)


# In[136]:


Y_pred_ridge = pipe.predict(X_test)
r2_score(Y_test, Y_pred_ridge)


# In[137]:


import pickle


# In[1]:


pickle.dump(pipe, open('RidgeModel.pkl','wb'))


# In[ ]:




