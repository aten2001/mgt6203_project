import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn import preprocessing, cluster
import sqlalchemy as sa
import numpy as np
import random
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


ENGINE_STR = 'mysql+pymysql://Jeffrey:AustinChrisSam@/PROJECT?host=mgt6203.c8s0bq6ntlv2.us-east-1.rds.amazonaws.com?port=3306'

sql = 'select * from consumer_analysis'
engine = sa.create_engine(ENGINE_STR)
soup_data = pd.read_sql_query(sql, engine)

full_soup_data = soup_data.copy()

# Remove response from dataframe
del (soup_data['Q1'])
del (soup_data['total_units_purchased'])
del (soup_data['total_units_purchased_on_mfr_coup'])
del (soup_data['total_units_purchased_on_store_coup'])
del (soup_data['male_head_avg_work_hours'])
del (soup_data['female_head_avg_work_hours'])
del (soup_data['rim_week'])
del (soup_data['household_id'])
del (soup_data['num_pets'])

# Code block taken from (with change a variable name)
# http://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
x = soup_data.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
soup_data_normalized = pd.DataFrame(x_scaled, columns=soup_data.columns).values

print(x_scaled.shape)

sample_indices = np.random.random_integers(0, 115254, int(np.round(x_scaled.shape[0] * 0.25)))
x_scaled_sample = x_scaled[sample_indices]

### Decide optimal k value for clustering

pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 500)

### Calinski-Harabaz Index
ks = range(2, 8)
for k in [2, 5, 8]:
    print('num clusters: ' + str(k))
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(soup_data_normalized)
    labels = kmeans_model.labels_

    cur_k = 'k'+str(k)
    full_soup_data[cur_k] = labels
    soup_data[cur_k] = labels
    min_k = soup_data[cur_k].min()
    max_k = soup_data[cur_k].max()
    for k in range(min_k, max_k + 1):
        print('k grouping: ' + str(k))
        cur_k_data = soup_data[soup_data[cur_k] == k]
        print(cur_k_data.describe())


# full_soup_data.to_sql('cluster_results', engine, if_exists='append', index=False)
# print('dataframe uploaded')


def elbow_analysis():
    # ### Elbow method
    KM = [KMeans(n_clusters=k).fit(x_scaled_sample) for k in ks]
    centroids = [k.cluster_centers_ for k in KM]
    D_k = [cdist(x_scaled_sample, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D, axis=1) for D in D_k]
    dist = [np.min(D, axis=1) for D in D_k]
    avgWithinSS = [sum(d) / x_scaled_sample.shape[0] for d in dist]
    # Total with-in sum of square
    wcss = [sum(d ** 2) for d in dist]
    tss = sum(pdist(x_scaled_sample) ** 2) / x_scaled_sample.shape[0]
    bss = tss - wcss
    kIdx = 10 - 1
    # Display Elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ks, avgWithinSS, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')


#elbow_analysis()