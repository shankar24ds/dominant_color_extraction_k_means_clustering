import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

def kmeans_clustering(uploaded_file, num):
    # to image dataset
    image_data = uploaded_file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    rgb_dataset = np.reshape(image_rgb, (height * width, 3))
    rgb_dataset = pd.DataFrame(rgb_dataset, columns=['red', 'green', 'blue'])
    # kmeans clustering
    kmeans = KMeans(n_clusters=num, init='k-means++', random_state=42, n_init='auto')
    kmeans.fit(rgb_dataset)
    cluster_labels = kmeans.labels_
    # labeled dataset
    rgb_dataset['cluster_label'] = cluster_labels
    rgb_dataset_grpby = rgb_dataset.groupby('cluster_label').agg(red=('red','mean'), green=('green', 'mean'), blue=('blue', 'mean'), proportion=('cluster_label', 'count')).reset_index(drop=True)
    rgb_dataset_grpby['relative_proportion'] = round((rgb_dataset_grpby['proportion']/sum(rgb_dataset_grpby['proportion']))*100, 2)
    rgb_dataset_grpby = rgb_dataset_grpby.sort_values('relative_proportion', ascending=False)
    rgb_dataset_grpby['red'] = rgb_dataset_grpby['red'].astype(int)
    rgb_dataset_grpby['green'] = rgb_dataset_grpby['green'].astype(int)
    rgb_dataset_grpby['blue'] = rgb_dataset_grpby['blue'].astype(int)
    rgb_dataset_grpby.drop(columns='proportion', inplace=True)
    # fig
    fig = px.pie(
        rgb_dataset_grpby,
        values="relative_proportion",
        names=[f'Color {i}' for i in range(len(rgb_dataset_grpby))],
        color_discrete_sequence=[f'rgb({row["red"]}, {row["green"]}, {row["blue"]})' for index, row in rgb_dataset_grpby.iterrows()],
        title="RGB Color Proportions"
    )
    return fig