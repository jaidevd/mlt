# coding: utf-8
import pandas as pd
df = pd.read_csv('tennis.csv')
df
y = df['play']
y
p = (y == 'yes').sum() / len(y)
p
get_ipython().run_line_magic('pylab', '')
log2
y_ent = -p * log2(p) - (1-p)*log2(1-p)
y_ent
# Splitting based on Wind
xdf = df[['wind', 'play']]
xdf
p_wind_strong = (xdf[xdf['wind'] == 'strong']['play'] == 'yes').sum() / xdf[xdf['wind'] == 'strong']
p_wind_strong = (xdf[xdf['wind'] == 'strong']['play'] == 'yes').sum() / xdf[xdf['wind'] == 'strong'].shape[0]
p_wind_strong
(xdf[xdf['wind'] == 'strong']['play'] == 'yes').sum()
p_wind_weak = (xdf[xdf['wind'] == 'weak']['play'] == 'yes').sum() / xdf[xdf['wind'] == 'weak'].shape[0]
xdf[xdf['wind'] == 'weak']
p_wind_weak
wind_weak_ent = -p_wind_weak * log2(p_wind_weak) + (1 - p_wind_weak) * log2(1 - p_wind_weak)
wind_weak_ent
wind_weak_ent = -p_wind_weak * log2(p_wind_weak) - (1 - p_wind_weak) * log2(1 - p_wind_weak)
wind_weak_ent
wind_weak_strong = -p_wind_strong * log2(p_wind_strong) - (1-  p_wind_strong) * log2(1 - p_wind_strong)
wind_weak_strong
# IG(S, Wind)?
y_ent
n_wind_weak = xdf[xdf['wind'] == 'weak'].sum()
n_wind_strong = xdf[xdf['wind'] == 'strong'].sum()
ig_wind = y_ent - (n_wind_weak / xdf.shape[0] * wind_weak_ent + n_wind_strong / xdf.shape[0] * wind_strong)
wind_strong_ent = -p_wind_strong * log2(p_wind_strong) - (1-  p_wind_strong) * log2(1 - p_wind_strong)
ig_wind = y_ent - (n_wind_weak / xdf.shape[0] * wind_weak_ent + n_wind_strong / xdf.shape[0] * wind_strong)
ig_wind = y_ent - (n_wind_weak / xdf.shape[0] * wind_weak_ent + n_wind_strong / xdf.shape[0] * wind_strong_ent)
n_wind_weak
n_wind_weak = xdf[xdf['wind'] == 'weak'].shape[0]
n_wind_strong = xdf[xdf['wind'] == 'strong'].shape[0]
ig_wind = y_ent - (n_wind_weak / xdf.shape[0] * wind_weak_ent + n_wind_strong / xdf.shape[0] * wind_strong_ent)
ig_wind
n_wind_weak
df
xdf = df[['outlook', 'play']]
xdf['outlook'].unique()
xdf['outlook'].value_counts()
n_sunny, n_rain, n_overcast = 5, 5, 4
p_sunny, p_rain, p_overcast = 5 / 14, 5/ 14, 4 / 14
def entropy(p):
    return -p * log2(p) - (1 - p) * log2(1 - p)
    
sunny_ent, rain_ent, overcast_ent = map(entropy, (p_sunny, p_rain, p_overcast))
ig_outlook = y_ent - sum([n_sunny / 14 * sunny_ent, n_rain / 14 * rain_ent, n_overcast / 14 * overcast_ent])
ig_outlook
ig_wind
sunny_ent
overcast_ent
xdf.groupby('outlook').agg(lambda x : (x['play'] == 'yes').sum() / x.shape[0])
xdf
xdf.groupby('outlook').agg(lambda x : (x == 'yes').sum() / x.shape[0])
xdf.groupby('outlook').agg(lambda x : (x == 'yes').sum() / x.shape[0])['play'].map(entropy)
get_ipython().run_line_magic('pinfo2', 'entropy')
log2(0)
xdf.groupby('outlook').agg(lambda x : (x == 'yes').sum() / x.shape[0])['play'].map(entropy).fillna(0)
outlook_ent = xdf.groupby('outlook').agg(lambda x : (x == 'yes').sum() / x.shape[0])['play'].map(entropy).fillna(0)
outlook_ent
df['outlook'].value_counts()
df['outlook'].value_counts() / 14
df['outlook'].value_counts() / 14 * outlook_ent
(df['outlook'].value_counts() / 14 * outlook_ent).sum()
ig_outlook = (df['outlook'].value_counts() / 14 * outlook_ent).sum()
ig_outlook
ig_outlook = y_ent - (df['outlook'].value_counts() / 14 * outlook_ent).sum()
ig_outlook
xdf = xdf[['temperature', 'play']]
xdf.head()
df
xdf = df[['temperature', 'play']]
temperature_ent = xdf.groupby('temperature').agg(lambda x: (x == 'yes').sum() / x.shape[0])['play'].map(entropy)
temperature_ent
ig_temp = y_ent - temperature_ent * xdf['temperature'].value_counts(normalize=True).sum()
ig_temp
temperature_ent * xdf['temperature'].value_counts(normalize=True)
temperature_ent * xdf['temperature'].value_counts(normalize=True).sum()
(temperature_ent * xdf['temperature'].value_counts(normalize=True)).sum()
ig_temp = y_ent - (temperature_ent * xdf['temperature'].value_counts(normalize=True)).sum()
ig_temp
xdf = df[['humidity', 'play']]
humidity_ent = xdf.groupby('humidity').agg(lambda x: (x == 'yes').sum() / x.shape[0])['play'].map(entropy)
humidity_ent
ig_humidity = y_ent - (humidity_ent * xdf['humidity'].value_counts(normalize=True)).sum()
ig_humidity
df
df.columns
ig_outlook
ig_temp
ig_humidity
ig_wind
