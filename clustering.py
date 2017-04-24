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
import pprint as pp


ENGINE_STR = 'mysql+pymysql://Jeffrey:AustinChrisSam@/PROJECT?host=mgt6203.c8s0bq6ntlv2.us-east-1.rds.amazonaws.com?port=3306'

# tables = ['mfg_consumer_analysis_quarterly', 'consumer_analysis_quarterly', 'mfg_consumer_analysis_total', 'consumer_analysis_total']
tables = ['consumer_analysis_quarterly', 'mfg_consumer_analysis_quarterly']

def elbow_analysis(table):
    # ### Elbow method
    KM = [KMeans(n_clusters=k).fit(soup_data_normalized_sample) for k in ks]
    centroids = [k.cluster_centers_ for k in KM]
    D_k = [cdist(soup_data_normalized_sample, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D, axis=1) for D in D_k]
    dist = [np.min(D, axis=1) for D in D_k]
    avgWithinSS = [sum(d) / soup_data_normalized_sample.shape[0] for d in dist]
    # Total with-in sum of square
    wcss = [sum(d ** 2) for d in dist]
    tss = sum(pdist(soup_data_normalized_sample) ** 2) / soup_data_normalized_sample.shape[0]
    bss = tss - wcss
    kIdx = 10 - 1
    # Display Elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ks, avgWithinSS, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for ' + table)
    plt.show()

for table in tables:
    print()
    print('table: ' + table)

    sql = 'select * from ' + table

    engine = sa.create_engine(ENGINE_STR)
    soup_data = pd.read_sql_query(sql, engine)

    full_soup_data = soup_data.copy()

    # Remove response from dataframe
    try:
        del (soup_data['Q1'])
        del (soup_data['Q2'])
        del (soup_data['Q3'])
        del (soup_data['Q4'])
        del (soup_data['total_units_purchased'])
        del (soup_data['total_units_purchased_on_mfr_coup'])
        del (soup_data['total_units_purchased_on_store_coup'])
        del (soup_data['rim_market'])
        del (soup_data['household_id'])
        del (soup_data['male_head_avg_work_hours'])
        del (soup_data['female_head_avg_work_hours'])
        del (soup_data['primary_head_avg_work_hours'])
        #del (soup_data['num_pets'])
        del (soup_data['weekday_shopper'])
        #del (soup_data['num_large_appliances'])
        del (soup_data['num_small_appliances'])
        del (soup_data['year'])
        del (soup_data['store_coup_bool'])
    except KeyError as ke:
        print('WARNING: KeyError')
        print(ke)

    try:
        del (soup_data['mfg_coup_bool'])
    except KeyError as ke:
        print('WARNING: KeyError')
        print(ke)

    print(soup_data.columns.values)

    # Code block taken from (with change a variable name)
    # http://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
    x = soup_data.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    soup_data_normalized = pd.DataFrame(x_scaled, columns=soup_data.columns).values

    sample_indices = np.random.random_integers(0, soup_data_normalized.shape[0]-1, int(np.round(soup_data_normalized.shape[0] * 0.25)))
    soup_data_normalized_sample = soup_data_normalized[sample_indices]

    ### Decide optimal k value for clustering

    pd.set_option('display.max_columns', 12)
    pd.set_option('display.width', 500)

    ### Calinski-Harabaz Index
    ks = range(2, 6)
    for k in ks:
        kmeans_model = KMeans(n_clusters=k, random_state=1).fit(x_scaled)
        labels = kmeans_model.labels_
        score = metrics.calinski_harabaz_score(x_scaled, labels)
        print('k: ' + str(k) + ' score: ' + str(score))

    elbow_analysis(table)

    print()

    print('non cluster summary')
    print(soup_data.describe())


    print('clustering summaries')


    ### Print summary statistics
    # """
    for k in [3]:
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
    
    ### Push results to database
    # full_soup_data.to_sql(table + '_cluster_results', engine, if_exists='append', index=False)
    # print('dataframe uploaded')
    # """



