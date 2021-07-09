import datetime
import glob
import gzip
import json
import logging

import pandas as pd

import src.mex.tower as mex_tower

# Develop after Study2_NewMobFeature -v2 @ 2021/02/04
# V2 count home based on tower point assigned municipality, which covers only 764 municipalities in 1397
# Count night tower here instead, so that I can assign new home to users after I figure out a better way
version = 'v1'
version_info = 'Top 4 most frequent night tower'
debug = False
debug_str = '-debug' if debug else ''

logging.basicConfig(filename=f"logs/Study2_MostFreqNightTower-{version}{debug_str}.log", level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info('===============================')
logging.info(f'Starting Study2_MostFreqNightTower {version}. {version_info}. Debug: {debug}')
print(f'Starting Study2_MostFreqNightTower {version}. {version_info}. Debug: {debug}')

tower_pts = mex_tower.pts()

id2gt = tower_pts.gtid.to_dict()

t2gt = {}  # tower id to group-tower id
t2gtid = {}  # tower id to numeric group-tower id (order as tower_pts)
for gtid, row in tower_pts.iterrows():
    for tid in row[0]:
        t2gt[tid] = row['gtid']
        t2gtid[tid] = gtid

for t, gt in t2gt.items():
    assert gt == id2gt[t2gtid[t]]


def add(d, k1, k2):
    if k1 not in d:
        d[k1] = {}

    if k2 not in d[k1]:
        d[k1][k2] = 0

    d[k1][k2] += 1


def iter_agg(agg):
    # for each date
    for tid, hr_uniq_users in agg.items():
        tid = tid.replace('33F430', '')
        # for each tower with location
        if tid in t2gt:
            gtid = t2gtid[tid]
            # for each hour 
            for hr, uniq_users in hr_uniq_users.items():
                # for each user
                for uid in uniq_users:
                    yield tid, gtid, hr, uid  # a user(uid) is at tid/gtid/munic at hour(hr)


def update_single_count(agg, count, kind='gtid', uid_set=None):
    """
    count: [user][gtid]=count
    """
    assert kind in ('all', 'night')

    for tid, gtid, hr, uid in iter_agg(agg):
        if uid_set is not None and uid not in uid_set:
            continue

        if kind == 'all':
            # add 1 to the frequency of being present at gtid (each hour at most once)
            add(count, uid, gtid)
        if kind == 'night' and hr in ['22', '23', '0', '1', '2', '3', '4', '5', '6']:
            # add 1 to the frequency of night time being at each municipality (each hour each tower at most once)
            add(count, uid, gtid)


fns = glob.glob('stats/AggMexTwDyHrUnqUsrVOZ/*.json.gz')
dates = sorted([fn.split('/')[-1][:-8] for fn in fns])

if debug: dates = dates[:2]

# Get user home municipalities
# [user][gtid]=count
night_tower_count = {}

start = datetime.datetime.now()
for i, date in enumerate(dates):
    path = f'stats/AggMexTwDyHrUnqUsrVOZ/{date}.json.gz'
    data = json.load(gzip.open(path))
    for subdate, agg in data.items():
        update_single_count(agg, night_tower_count, kind='night')
    time_spent = (datetime.datetime.now() - start).total_seconds()
    print(date, time_spent, len(night_tower_count))
    if (i + 1) % 30 == 0:
        logging.info(f'{date}, {time_spent} seconds, total_users: {len(night_tower_count)}')

user_top4_night_tower = []
for uid, night_tower in night_tower_count.items():

    top4_tower = sorted(night_tower.items(), key=lambda x: x[1], reverse=True)[:4]

    ures = {'uid': uid, 'n_night_tower': len(night_tower), 'night_presence': sum(night_tower.values())}
    for i, tcnt in enumerate(top4_tower):
        ures[f't{i + 1}'] = tcnt[0]
        ures[f'c{i + 1}'] = tcnt[1]
    user_top4_night_tower.append(ures)

pd.DataFrame(user_top4_night_tower).to_csv(f'data/Study2_individual_home_tower-{version}{debug_str}.csv.gz')
print('Done')
logging.info('Done')
