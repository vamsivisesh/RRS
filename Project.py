#!/usr/bin/env python
# coding: utf-8

# In[94]:
from flask import Flask, jsonify,render_template,url_for


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("always")
from flask import Flask,render_template,url_for
from flask import request


# In[95]:


data=pd.read_csv("HyderabadResturants.csv")
data.drop('links', axis=1, inplace = True)
data.head()

app=Flask(__name__)


# In[96]:


data.info()


# In[97]:


data.shape


# In[98]:


data[["cuisine1", "cuisine2",'cuisine3','cuisine4' ,'cuisine5','cuisine6','cuisine7','cuisine8']] = (  
    data["cuisine"].str.split(",", expand=True)     
)
data.rename(columns ={'price for one': 'price'},inplace=True)
data.head()





features=["cuisine1","cuisine2","cuisine3","cuisine4","cuisine5","cuisine6","cuisine7","cuisine8"]
data["temp"]=data[features].isnull().sum(axis=1)
data["no_Of_cusines"]=8-data["temp"]
data.head()






feature = data["cuisine"].tolist()
tfidf = text.TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)





indices = pd.Series(data.index, index=data['names']).drop_duplicates()





def restaurant_recommendation(name, similarity = similarity):
    index = indices[name]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    restaurantindices = [i[0] for i in similarity_scores]
    return data['names'].iloc[restaurantindices]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/index', methods=['POST'])
def keywords():
    output=request.form.get('output')
    return render_template( "res.html", rest=list(restaurant_recommendation(output).to_dict().values()) )

@app.route('/res',methods=['POST'])
def res():
    data=request.form.get('output')
    return render_template("res.html",rest=data)


if __name__=='__main__':
    app.run(debug=True)
    app.run()






