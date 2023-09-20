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
from flask_sqlalchemy import SQLAlchemy
import psutil
import sys
from flask import request
from Project.py import restaurant_recommendation

app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///test.db'
db=SQLAlchemy(app)

data=pd.read_csv("HyderabadResturants.csv")
data.drop('links', axis=1, inplace = True)

data.head()
data.info() 
data.shape

@app.route('/')
def index():
    data=restaurant_recommendation(food_item)
    return render_template('search.html',data=data)

@app.route('/res')
def res():
    return render_template('res.html')


# @app.route('/index', methods=['POST'])
# # def keywords():
# #     output=request.form.get('output')
# #     print(output)
# #     print(type(output))
# #     data[["cuisine1", "cuisine2",'cuisine3','cuisine4' ,'cuisine5','cuisine6','cuisine7','cuisine8']] = (  
# #     data["cuisine"].str.split(",", expand=True)     
# #     )
# #     data.rename(columns ={'price for one': 'price'},inplace=True)
# #     features=["cuisine1","cuisine2","cuisine3","cuisine4","cuisine5","cuisine6","cuisine7","cuisine8"]
# #     data["temp"]=data[features].isnull().sum(axis=1)
# #     data["no_Of_cusines"]=8-data["temp"]
# #     feature = data["cuisine"].tolist()
# #     tfidf = text.TfidfVectorizer()
# #     tfidf_matrix = tfidf.fit_transform(feature)
# #     similarity = cosine_similarity(tfidf_matrix)
# #     indices = pd.Series(data.index, index=data['names']).drop_duplicates()
# #     print('outside_indices',indices)
    
# #     def restaurant_recommendation(name,similarity = similarity):
# #         print('indices',indices)
# #         index = indices[name]
# #         similarity_scores = list(enumerate(similarity[index]))
# #         print(similarity_scores)
# #         similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
# #         similarity_scores = similarity_scores[0:10]
# #         restaurantindices = [i[0] for i in similarity_scores]
# #         return data['names'].iloc[restaurantindices]

# #     result=restaurant_recommendation(output)
# #     # print(result)
# #     # print(restaurant_recommendation("Subway"))
# #     # print(type(result))
# #     return render_template('index.html',keywords=result.to_html())

if __name__=='__main__':
    app.run(debug=True)
    app.run()

def api_response():
    from flask import jsonify
    if request.method == 'POST':
        return jsonify(**request.json)


