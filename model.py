import pandas as pd
import sqlalchemy as sa

ENGINE_STR = 'mysql+pymysql://Jeffrey:AustinChrisSam@/PROJECT?host=mgt6203.c8s0bq6ntlv2.us-east-1.rds.amazonaws.com?port=3306'

sql = 'select * from mfg_consumer_analysis_quarterly'
sql2 = 'select * from consumer_analysis_quarterly'
sql3 = 'select * from mfg_consumer_analysis_quarterly_cluster_results'
sql4 = 'select * from consumer_analysis_quarterly_cluster_results'

engine = sa.create_engine(ENGINE_STR)
df = pd.read_sql_query(sql, engine)
df2 = pd.read_sql_query(sql2, engine)
df3 = pd.read_sql_query(sql3, engine)
df4 = pd.read_sql_query(sql4, engine)

print(df.head())


#df = pd.read_csv('../PROJECT_work_income_data.csv')
# print(df.head())

#df['week'] = df['rim_week'].apply(lambda x: str(x)[-2:])

"""
def get_quarter(date):
    week = date % 100
    if week <= 13:
        return 'Q1'
    elif 13 < week <= 26:
        return 'Q2'
    elif 26 < week <= 39:
        return 'Q3'
    else:
        return 'Q4'
"""

#df['quarter'] = df['rim_week'].apply(lambda x: get_quarter(x))
df.to_csv('mfg_consumer_analysis_quarterly.csv')
df2.to_csv('consumer_analysis_quarterly.csv')
df3.to_csv('mfg_consumer_analysis_quarterly_cluster_results.csv')
df4.to_csv('consumer_analysis_quarterly_cluster_results')