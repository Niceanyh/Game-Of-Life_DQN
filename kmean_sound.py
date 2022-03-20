from sklearn.cluster import KMeans
import numpy as np
# save the model
import pickle


# load the model
# model = pickle.load(open("model.pkl", "rb"))

def train(x):
    kmeans = KMeans(n_clusters=21, random_state=0).fit(x)
    pickle.dump(kmeans, open("kmeans_sound.pkl", "wb"))