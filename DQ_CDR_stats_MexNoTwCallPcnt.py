import datetime
import glob
import gzip
import json
import os

import matplotlib
import pandas as pd

matplotlib.use('agg')


def sum_count(d):
    a = 0
    nt1 = 0
    nt2 = 0
    nt12 = 0
    for k, v in d.items():
        a += v.get('all', 0)
        nt1 += v.get('nt1', 0)
        nt2 += v.get('nt2', 0)
        nt12 += v.get('nt12', 0)
    return a, nt1, nt2, nt12


sd = datetime.datetime.now()
print('stated', sd)
voz_or_en = 'VOZ'
stats_dir = 'stats/DQAggMexUsrNoTwCall%s/' % voz_or_en

fns = sorted(list(glob.glob(stats_dir + '*.json.gz')))
num_calls = {}
for i, fn in enumerate(fns):
    if i % 10 == 0:
        print('working on %dth file: %s' % (i, fn))
    date = os.path.basename(fn).replace('.json.gz', '')
    with gzip.open(fn) as fin:
        data = json.load(fin)
        num_calls[date] = sum_count(data)

df = pd.DataFrame(list(num_calls.values()), index=list(num_calls.keys()), columns=['all', 'nt1', 'nt2', 'nt12'])
df['nt1pct'] = df.nt1 / df['all']
df['nt2pct'] = df.nt2 / df['all']
df['nt12pct'] = df.nt12 / df['all']
nt1p = 100 * df.nt1.sum() / df['all'].sum()
nt2p = 100 * df.nt2.sum() / df['all'].sum()
nt12p = 100 * df.nt12.sum() / df['all'].sum()
df.to_csv('stats/MexNoTwCallPcntDaily.csv')

ax = df[['nt1pct', 'nt2pct', 'nt12pct']].plot.hist(
    alpha=0.5, title='overall no t1=%0.2f%% and no t2=%0.2f%% and no t1&2=%0.2f%%' % (nt1p, nt2p, nt12p))
fig = ax.get_figure()
fig.savefig('stats/MexNoTwCallPcntDailyHist.png')

print('ed', datetime.datetime.now(), datetime.datetime.now() - sd)
