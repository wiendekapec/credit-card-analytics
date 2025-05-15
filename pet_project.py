# pet_project.py
# End‑to‑End BI‑демо‑проект: синтетический CSV → PostgreSQL → Star Schema → аналитика

import os, pandas as pd, numpy as np
from io import StringIO
from sqlalchemy import create_engine
from pandas.api.types import (
    is_datetime64_any_dtype, is_float_dtype, is_integer_dtype
)

# ─────────── 1. параметры подключения ───────────
DB = dict(
    dbname   = 'postgres',          # <-- замените на своё
    user     = 'postgres',
    password = 'postgres',
    host     = 'localhost',
    port     = 5432,
)

CSV_PATH  = 'creditcard.csv'      # синтетический файл
engine    = create_engine(
    f"postgresql://{DB['user']}:{DB['password']}@{DB['host']}:{DB['port']}/{DB['dbname']}"
)
raw_conn  = engine.raw_connection()
cur       = raw_conn.cursor()

# ─────────── 2. читаем CSV ───────────
df = pd.read_csv(CSV_PATH, parse_dates=['timestamp'])
print(f"Загружено {len(df):,} строк из CSV")

# ─── приведение имён колонок к «универсальным» ───
df = df.rename(columns={
    'amount'   : 'Amount',       # с большой буквы (для совместимости ниже)
    'is_fraud' : 'Class'         # 0 / 1
})
df['actual_ts'] = pd.to_datetime(df['timestamp'])


# ─────────── 3. staging в PostgreSQL ───────────
def pg_type(dtype):
    if is_datetime64_any_dtype(dtype): return 'TIMESTAMP'
    if is_integer_dtype(dtype):        return 'BIGINT'
    if is_float_dtype(dtype):          return 'DOUBLE PRECISION'
    return 'TEXT'
stg = 'stg_creditcard'
cols_sql = ', '.join(f'{c} {pg_type(t)}' for c, t in df.dtypes.items())

cur.execute(f'DROP TABLE IF EXISTS {stg} CASCADE;')   # всегда пересоздаём
cur.execute(f'CREATE TABLE {stg} ({cols_sql});')
raw_conn.commit()

buf = StringIO(); df.to_csv(buf, index=False, header=False); buf.seek(0)
cur.copy_expert(f'COPY {stg} FROM STDIN WITH (FORMAT csv)', buf)
raw_conn.commit()
print('Staging loaded.')


# ─────────── 4. датафрейм из staging ───────────
stg_df = df.copy()                            # у нас всё уже в pandas
stg_df['date'] = stg_df['actual_ts'].dt.normalize()
stg_df['bucket_raw'], bins = pd.qcut(
    stg_df['Amount'], 5, labels=False, retbins=True, duplicates='drop'
)

# ─────────── 5. размерности ───────────
dim_date = (stg_df[['date']]
            .drop_duplicates().reset_index(drop=True)
            .assign(date_id = lambda d: d.index+1,
                    day     = lambda d: d['date'].dt.day,
                    month   = lambda d: d['date'].dt.month,
                    year    = lambda d: d['date'].dt.year,
                    weekday = lambda d: d['date'].dt.dayofweek))

dim_amount = (pd.DataFrame({
        'bucket_raw': range(len(bins)-1),
        'min_amount': bins[:-1],
        'max_amount': bins[1:]
    }).assign(bucket_id = lambda x: x.index+1))

dim_class = pd.DataFrame({'class_id':[0,1], 'class_desc':['legit','fraud']})

def load_dim(name, df_dim, cols, ddl):
    cur.execute(ddl)
    cur.execute(f'TRUNCATE TABLE {name} CASCADE;')
    raw_conn.commit()
    buf = StringIO(); df_dim[cols].to_csv(buf, index=False, header=False); buf.seek(0)
    cur.copy_expert(f'COPY {name} FROM STDIN WITH (FORMAT csv)', buf)
    raw_conn.commit()

load_dim('dim_date', dim_date,
         ['date_id','date','day','month','year','weekday'],
         'CREATE TABLE IF NOT EXISTS dim_date('
         'date_id INT PRIMARY KEY, date DATE, day INT, month INT, year INT, weekday INT);')

load_dim('dim_amount_bucket', dim_amount,
         ['bucket_id','min_amount','max_amount'],
         'CREATE TABLE IF NOT EXISTS dim_amount_bucket('
         'bucket_id INT PRIMARY KEY, min_amount DOUBLE PRECISION, max_amount DOUBLE PRECISION);')

load_dim('dim_class', dim_class,
         ['class_id','class_desc'],
         'CREATE TABLE IF NOT EXISTS dim_class('
         'class_id INT PRIMARY KEY, class_desc TEXT);')

print('Dimensions loaded.')

# ─────────── 6. факт‑таблица ───────────
cur.execute('CREATE TABLE IF NOT EXISTS fact_transactions('
            'fact_id SERIAL PRIMARY KEY,'
            'fk_date_id INT REFERENCES dim_date(date_id),'
            'fk_amount_bucket_id INT REFERENCES dim_amount_bucket(bucket_id),'
            'fk_class_id INT REFERENCES dim_class(class_id),'
            'amount DOUBLE PRECISION);')
cur.execute('TRUNCATE TABLE fact_transactions;'); raw_conn.commit()

fact_df = (stg_df
           .merge(dim_date[['date','date_id']], on='date')
           .merge(dim_amount[['bucket_raw','bucket_id']], on='bucket_raw')
           .rename(columns={
               'date_id'  :'fk_date_id',
               'bucket_id':'fk_amount_bucket_id',
               'Class'    :'fk_class_id',
               'Amount'   :'amount'})
          )[['fk_date_id','fk_amount_bucket_id','fk_class_id','amount']]
fact_df = fact_df.astype({'fk_date_id':int,'fk_amount_bucket_id':int,'fk_class_id':int})

buf = StringIO(); fact_df.to_csv(buf, index=False, header=False); buf.seek(0)
cur.copy_expert(
    'COPY fact_transactions (fk_date_id,fk_amount_bucket_id,fk_class_id,amount) '
    'FROM STDIN WITH (FORMAT csv)', buf)
raw_conn.commit()
print('Star schema loaded.')

# ─────────── 7. пример аналитики ───────────
top5 = pd.read_sql(
    'WITH t AS (SELECT fk_date_id, amount, '
    'ROW_NUMBER() OVER(PARTITION BY fk_date_id ORDER BY amount DESC) rn '
    'FROM fact_transactions) '
    'SELECT * FROM t WHERE rn<=5;', engine)
print('Top‑5 транзакций в день:\n', top5.head())

# ─────────── 8. RFM‑анализ (по customer_id) ───────────
now = stg_df['actual_ts'].max() + pd.Timedelta(days=1)
rfm = (stg_df
       .groupby('customer_id')
       .agg(last=('actual_ts','max'),
            frequency=('customer_id','size'),
            monetary=('Amount','sum')))
rfm['recency'] = (now - rfm['last']).dt.days

# безопасная сегментация на квантильные группы
def quantile_bucket(series, n=5, ascending=True):
    ranks = series.rank(pct=True)
    buckets = (ranks * n).apply(np.ceil).astype(int).clip(1, n)
    return buckets if ascending else n - buckets + 1

rfm['R'] = quantile_bucket(rfm['recency'], ascending=False)
rfm['F'] = quantile_bucket(rfm['frequency'])
rfm['M'] = quantile_bucket(rfm['monetary'])
rfm['RFM_Segment'] = rfm['R'].astype(str)+rfm['F'].astype(str)+rfm['M'].astype(str)

print("RFM sample (by synthetic customer_id):\n", rfm.head())

# ─────────── 9. кластеризация ───────────
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = rfm[['recency','frequency','monetary']]
X_scaled = StandardScaler().fit_transform(X)
k = min(5, len(rfm))                # ≤ 5 кластеров, но не больше клиентов
if k >= 2:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    rfm['cluster'] = km.fit_predict(X_scaled)
    print("Clusters distribution:\n", rfm['cluster'].value_counts())

# ─────────── 10. когортный анализ ───────────
stg_df['cohort_month'] = (
    stg_df.groupby('customer_id')['actual_ts'].transform('min').dt.to_period('M')
)
stg_df['order_month'] = stg_df['actual_ts'].dt.to_period('M')
cohort = (stg_df.groupby(['cohort_month','order_month'])
          .size().unstack(fill_value=0))

# ─────────── 11. A/B‑тест (по клиентам) ───────────
rng = np.random.default_rng(42)
cust_ids = rfm.index.to_numpy()
group_A = set(rng.choice(cust_ids, size=len(cust_ids)//2, replace=False))
fact_ab = fact_df.copy()
fact_ab['group'] = fact_ab['fk_class_id'].apply(
    lambda x: 'A' if x in group_A else 'B')
ab_stats = fact_ab.groupby('group')['amount'].agg(['count','mean','var'])
if {'A','B'} <= set(ab_stats.index):
    n1,n2 = ab_stats.loc['A','count'], ab_stats.loc['B','count']
    m1,m2 = ab_stats.loc['A','mean'],  ab_stats.loc['B','mean']
    v1,v2 = ab_stats.loc['A','var'],   ab_stats.loc['B','var']
    se = np.sqrt(v1/n1 + v2/n2)
    z  = (m1 - m2) / se
    from scipy import stats
    p  = 2*(1 - stats.norm.cdf(abs(z)))
    ci = stats.norm.interval(0.95, loc=m1-m2, scale=se)
    print(f"A/B test:  A={m1:.2f}  B={m2:.2f}  z={z:.3f}  p={p:.4f}  CI{ci}")

# ─────────── 12. визуализация (урезанная, чтобы скрипт не открывал окна при cron‑запуске) ───────────
import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.figure(); rfm['recency'].hist(bins=30); plt.title('Recency'); plt.xlabel('Days'); plt.ylabel('Customers')
plt.figure(); rfm['frequency'].hist(bins=30); plt.title('Frequency'); plt.xlabel('Txns'); plt.ylabel('Customers')
plt.figure(); (rfm['monetary'].clip(upper=rfm['monetary'].quantile(0.99))
               .hist(bins=30)); plt.title('Monetary (≤99‑perc)'); plt.xlabel('Spend'); plt.ylabel('Customers')
plt.show()

# ─────────── завершение ───────────
cur.close(); raw_conn.close()
print('Pipeline complete.')