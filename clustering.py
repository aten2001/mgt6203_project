import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn import preprocessing, cluster
import sqlalchemy as sa
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

ENGINE_STR = 'mysql+pymysql://Jeffrey:AustinChrisSam@/PROJECT?host=mgt6203.c8s0bq6ntlv2.us-east-1.rds.amazonaws.com?port=3306'

sql = 'select * from consumer_analysis'
engine = sa.create_engine(ENGINE_STR)
soup_data = pd.read_sql_query(sql, engine)

# remove response from dataframe
# del (soup_data['total_units_purchased'])
# del (soup_data['total_units_purchased_on_mfr_coup'])
# del (soup_data['total_units_purchased_on_store_coup'])


# response_df = soup_data[['total_units_purchased',
#                          'total_units_purchased_on_mfr_coup',
#                          'total_units_purchased_on_store_coup']]

# response_df = soup_data[['total_units_purchased',
#                          'total_units_purchased_on_mfr_coup']]

response_df = soup_data[['total_units_purchased',
                         'total_units_purchased_on_store_coup']]

# Code block taken from (with change a variable name)
# http://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
x = response_df.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
soup_data_normalized = pd.DataFrame(x_scaled, columns=response_df.columns)

"""
kmeans_3 = cluster.KMeans(n_clusters=3)
kmeans_4 = cluster.KMeans(n_clusters=4)
kmeans_5 = cluster.KMeans(n_clusters=5)
kmeans_6 = cluster.KMeans(n_clusters=6)
kmeans_7 = cluster.KMeans(n_clusters=7)

k_3_clusters = kmeans_3.fit_predict(soup_data_normalized)
k_4_clusters = kmeans_4.fit_predict(soup_data_normalized)
k_5_clusters = kmeans_5.fit_predict(soup_data_normalized)
k_6_clusters = kmeans_6.fit_predict(soup_data_normalized)
k_7_clusters = kmeans_7.fit_predict(soup_data_normalized)
"""


######################################
# Code between lines of # signs taken from
# https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return (mu, clusters)


def cluster_points(X, mu):
    clusters = dict((tuple(m), []) for m in np.array(mu))
    for x in X:
        best_mu = mu[0]
        best_mu_dist = np.linalg.norm(np.subtract(x, mu[0]).tolist())
        for i in mu:
            mu_dist = np.linalg.norm(np.subtract(x, i).tolist())
            if mu_dist < best_mu_dist:
                best_mu = i
                best_mu_dist = mu_dist
        clusters[tuple(best_mu)].append(x)
    return clusters


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis=0))
    return newmu


def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))


######################################



def init_board_gauss(N, k):
    n = float(N) / k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05, 0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a, b])
        X.extend(x)
    X = np.array(X)[:N]
    return X


######################################
# Code between lines of # signs taken from
# https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i] - list(c)) ** 2 / (2 * len(list(c))) \
                for i in range(K) for c in clusters[i]])


def bounding_box(X):
    xmin, xmax = min(X, key=lambda a: a[0])[0], max(X, key=lambda a: a[0])[0]
    ymin, ymax = min(X, key=lambda a: a[1])[1], max(X, key=lambda a: a[1])[1]
    return (xmin, xmax), (ymin, ymax)


def gap_statistic(X):
    (xmin, xmax), (ymin, ymax) = bounding_box(X)
    # Dispersion for real distribution
    ks = range(1, 10)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, clusters = find_centers(X, k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin, xmax),
                           random.uniform(ymin, ymax)])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb, k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs) / B
        sk[indk] = np.sqrt(sum((BWkbs - Wkbs[indk]) ** 2) / B)
    sk = sk * np.sqrt(1 + 1 / B)
    return (ks, Wks, Wkbs, sk)


# X = init_board_gauss(200, 3)
# ks, logWks, logWkbs, sk = gap_statistic(X.tolist())
######################################


### Calinski-Harabaz Index
ks = range(2, 15)
for k in ks:
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(x_scaled)
    labels = kmeans_model.labels_
    score = metrics.calinski_harabaz_score(x_scaled, labels)
    print('k: ' + str(k) + ' score: ' + str(score))

### Elbow method
KM = [KMeans(n_clusters=k).fit(x_scaled) for k in ks]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(x_scaled, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/x_scaled.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(x_scaled)**2)/x_scaled.shape[0]
bss = tss-wcss

kIdx = 10-1

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ks, avgWithinSS, 'b*-')
ax.plot(ks[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r',
        markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')
plt.show()