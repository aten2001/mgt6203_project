import pandas as pd
import sqlalchemy as sa

ENGINE_STR = 'mysql+pymysql://Jeffrey:AustinChrisSam@/PROJECT?host=mgt6203.c8s0bq6ntlv2.us-east-1.rds.amazonaws.com?port=3306'

sql = 'select * from upc_data'
engine = sa.create_engine(ENGINE_STR)
df = pd.read_sql_query(sql, engine)
print(df.head())
