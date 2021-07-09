# coding: utf-8

import datetime
import gc
import glob
import gzip
import json
import logging
from collections import defaultdict

import geopandas as gp
import numpy as np
import pandas as pd

import src.mex.regions2010 as mex_region
import src.mex.tower as mex_tower

version = 'v2'
version_info = 'Do not filter user by thres in this script'
debug = False
debug_str = '-debug' if debug else ''

logging.basicConfig(filename=f"logs/Study2_NewMobFeature-{version}{debug_str}.log", level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info('===============================')
logging.info(f'Starting Study2_NewMobFeatures {version}. {version_info}. Debug: {debug}')
print(f'Starting Study2_NewMobFeatures {version}. {version_info}. Debug: {debug}')

tower_vor = mex_tower.voronoi()

perimeter = tower_vor.geometry.apply(lambda x: x.length / 1000).copy()
perimeter.name = 'perimeter'

# high density -> small perimeter -> lower rate
# min perimeter -> c= 0.5

d = perimeter.values
c = (d - min(d)) / (max(d) - min(d))
a = 0.7
b = 1.3
c_ = np.log10((10 ** b - 10 ** a) * c + 10 ** a)

perimeter = perimeter.reset_index()
perimeter['c_factor'] = c_

tower_pts = mex_tower.pts()
tower_pts = tower_pts.merge(perimeter)
mgm = mex_region.municipalities()
t2mgm = gp.sjoin(tower_pts, mgm)
t2mgm = t2mgm.sort_index()

# In[36]:


id2gt = t2mgm.gtid.to_dict()

t2gt = {}
t2gtid = {}
t2munic = {}
gt2munic = {}
for gtid, row in t2mgm.iterrows():
    gt2munic[gtid] = row['index_right']
    for tid in row[0]:
        t2gt[tid] = row['gtid']
        t2gtid[tid] = gtid
        t2munic[tid] = row['index_right']

for t, gt in t2gt.items():
    assert gt == id2gt[t2gtid[t]]

# In[8]:


fns = glob.glob('stats/AggMexTwDyHrUnqUsrVOZ/*.json.gz')

dates = sorted([fn.split('/')[-1][:-8] for fn in fns])
if debug:
    dates = dates[:3]


# # count user's presents in municipalities and towers

def add(d, k1, k2):
    if k1 not in d:
        d[k1] = {}

    if k2 not in d[k1]:
        d[k1][k2] = 0

    d[k1][k2] += 1


# ## count users with night time activities first
# 
# filter users with at least 1 night activities first
# then count activities through out the day

# In[11]:


def iter_agg(agg):
    # for each date
    for tid, hr_uniq_users in agg.items():
        tid = tid.replace('33F430', '')
        # for each tower with location
        if tid in t2gt:
            gtid = t2gtid[tid]
            munic_id = t2munic[tid]
            # for each hour 
            for hr, uniq_users in hr_uniq_users.items():
                # for each user
                for uid in uniq_users:
                    yield tid, gtid, munic_id, hr, uid  # a user(uid) is at tid/gtid/munic at hour(hr)


def update_single_count(agg, count, kind='gtid', uid_set=None):
    """
    count: [user][gtid/munic_id]=count
    """
    assert kind in ('gtid', 'home')

    for tid, gtid, munic_id, hr, uid in iter_agg(agg):
        if uid_set is not None and uid not in uid_set:
            continue

        if kind == 'gtid':
            # add 1 to the frequency of being present at gtid (each hour at most once)
            add(count, uid, gtid)
        if kind == 'home' and hr in ['22', '23', '0', '1', '2', '3', '4', '5', '6']:
            # add 1 to the frequency of night time being at each municipality (each hour each tower at most once)
            add(count, uid, munic_id)


# =======================================
# Get user home municipalities
print('getting user home municipalities')
logging.info('getting user home municipalities')
night_mun_count = {}  # [user][munic_id]=count
start = datetime.datetime.now()
for i, date in enumerate(dates):
    path = f'stats/AggMexTwDyHrUnqUsrVOZ/{date}.json.gz'
    data = json.load(gzip.open(path))
    for subdate, agg in data.items():
        update_single_count(agg, night_mun_count, kind='home')
    time_spent = (datetime.datetime.now() - start).total_seconds()
    print(date, time_spent, len(night_mun_count))
    if (i + 1) % 30 == 0:
        logging.info(f'{date}, {time_spent} seconds, total_users: {len(night_mun_count)}')
end = datetime.datetime.now()
time_spent = (end - start).total_seconds()
logging.info(f'Total time spend on night_mun_count {time_spent} seconds')
print(f'Total time spend on night_mun_count {time_spent} seconds')

uhome = {uid: max(stats, key=stats.get) for uid, stats in night_mun_count.items()}
uhome = pd.Series(uhome).reset_index()
uhome.columns = ['uid', 'home_munic']
logging.info(f'n_user with night activity: {len(uhome)}, '
             f'n_munic is home: {uhome.home_munic.nunique()}')

# save memory
del night_mun_count
gc.collect()

# =======================================
# Get user activity count at tower level
# this is not done at the same time as above to save memory.
# for users with at least 1 night activity
print('Get user activity count at tower level')
logging.info('Get user activity count at tower level')
activity_count = {}  # [user][gtid]=count
start = datetime.datetime.now()
for i, date in enumerate(dates):
    path = f'stats/AggMexTwDyHrUnqUsrVOZ/{date}.json.gz'
    data = json.load(gzip.open(path))
    for subdate, agg in data.items():
        update_single_count(agg, activity_count, kind='gtid',
                            uid_set=set(uhome.uid.tolist()))
    print(date, (datetime.datetime.now() - start).total_seconds(), len(activity_count))
    if (i + 1) % 30 == 0:
        logging.info(f'{date}, {time_spent} seconds')
end = datetime.datetime.now()
time_spent = (end - start).total_seconds()
logging.info(f'Total time spend on activity_count: {time_spent} seconds')
print(f'Total time spend on activity_count: {time_spent} seconds')

# In[23]:


uhome['total_presence'] = uhome.uid.apply(lambda x: sum(activity_count[x].values()) if x in activity_count else 0)
uhome.set_index('uid', inplace=True)

# =========================================
# # compute features
logging.info('...Compute Features')


# TODO:
#  Maybe, RoG and entropy can be computed at daytime vs night time

def get_mob_ftr(u_tower_count):
    u_gtids = list(u_tower_count.keys())
    u_gt_coord = np.array(t2mgm.loc[u_gtids].geometry.apply(lambda x: x.centroid.coords[0]).tolist())
    u_c_factor = t2mgm.loc[u_gtids, 'c_factor'].values
    u_freq = np.array(list(u_tower_count.values()))
    u_freq_vec = u_freq[:, np.newaxis]

    # center of mass (i.e., the weighted mean point of thephone towers visited by an individual)
    center_mass = (u_gt_coord * u_freq_vec).sum(axis=0) / u_freq_vec.sum()
    ri_rcm_2 = ((u_gt_coord - center_mass) ** 2).sum(axis=1)
    ri_rcm_2_freq = ri_rcm_2 * u_freq
    rog = ri_rcm_2_freq.sum() / sum(u_freq)
    rog = np.sqrt(rog)  # in meters

    pi = u_freq / u_freq.sum()
    entropy = -sum(pi * np.log(pi))
    correct_entropy = -sum(pi * np.log(pi) * u_c_factor)
    return rog, entropy, correct_entropy


# ----------------------------------------
print('start on mob features')
print_freq = 1e+3 if debug else 1e+5
start = datetime.datetime.now()
user_mob_feature = []
for i, uid in enumerate(uhome.index):
    if (i + 1) % print_freq == 0:
        print('working on mob features', i)
        if debug: break
    if (i + 1) % (5 * print_freq) == 0:
        logging.info(f'working on mob features {i}')
    if uid not in activity_count:  # I think this is not necessary
        logging.info(f'{uid} not in activity count')
        continue
    rog, ent, cent = get_mob_ftr(activity_count[uid])
    user_mob_feature.append({'uid': uid, 'RoG': rog,
                             'Entropy': ent,
                             'CorrectEntropy': cent})

user_mob_feature = pd.DataFrame(user_mob_feature)
user_mob_feature.set_index('uid', inplace=True)
end = datetime.datetime.now()
time_spent = (end - start).total_seconds()
logging.info(f'Total time spend on get_mob_ftr: {time_spent} seconds')
print(f'Total time spend on get_mob_ftr: {time_spent} seconds')

# -----------------------------
print('start on user_home_vs_nhome')
start = datetime.datetime.now()
print_freq = 1e+3 if debug else 1e+5
user_home_vs_nhome = []
for i, uid in enumerate(uhome.index):
    if (i + 1) % print_freq == 0:
        print('working on home vs non-home presence', i)
        if debug: break
    if (i + 1) % (5 * print_freq) == 0:
        logging.info(f'working on home vs non-home presence {i}')
    if uid not in activity_count:  # I think this is not necessary
        logging.info(f'{uid} not in activity count')
        continue
    home_munic_count = 0
    non_home_count = 0
    home_munic_u = uhome.loc[uid].home_munic
    for gtid, count in activity_count[uid].items():
        # user *uid* appear at tower *gtid* *count* times
        current_munic_u = gt2munic[gtid]
        if current_munic_u == home_munic_u:
            home_munic_count += count
        else:
            non_home_count += count
    user_home_vs_nhome.append({'uid': uid, 'home_presence': home_munic_count, 'non_home_presence': non_home_count})

user_home_vs_nhome = pd.DataFrame(user_home_vs_nhome).set_index('uid')
end = datetime.datetime.now()
time_spent = (end - start).total_seconds()
logging.info(f'Total time spend on user_home_vs_nhome: {time_spent} seconds')
print(f'Total time spend on user_home_vs_nhome: {time_spent} seconds')

# saving individual features
uhome.join(user_home_vs_nhome, how='outer') \
    .join(user_mob_feature, how='outer') \
    .to_csv(f'data/Study2_individual_features-{version}{debug_str}.csv.gz')
# save memory
del user_home_vs_nhome, user_mob_feature
gc.collect()

# ------------------------------------
print('start on municipality flow')
# in flow: groupby src,
# - n_in_munic: count n_unique dst,
# - n_in_users: sum n_unique users of different dst (no duplicates because users has one unique home)
# - n_in_activity: sum count of different dst
# - std of activity over *dst*; std of n_users over *dst*
# out flow: groupby dst, the rest replace dst in above with src.
# same definition of OD as SafeGraph Covid Dataset.
# if src==dst, not sure if self-loop is necessary, but keep it anyway in case it could be useful.

# munic level, {(src, dst): total presence at *dst* from all user with home at *src*}
munic_flow_count = defaultdict(int)
# munic level, {(src, dst): unique users whose home at *src* has been to *dst*}
munic_flow_unique_user = defaultdict(set)

start = datetime.datetime.now()
print_freq = 1e+3 if debug else 1e+5
for i, uid in enumerate(uhome.index):
    if (i + 1) % print_freq == 0:
        print('working on municipality flow', i)
        if debug: break
    if (i + 1) % (5 * print_freq) == 0:
        logging.info(f'working on municipality flow {i}')

    home_munic_u = uhome.loc[uid].home_munic
    total_presence = uhome.loc[uid].total_presence
    for gtid, count in activity_count[uid].items():
        # user *uid* appear at tower *gtid* *count* times
        current_munic_u = gt2munic[gtid]
        # it's ok if current==home, it is self-loop.
        # TODO: cannot loop threshold. run extremely long and got killed supposedly bc. out-of-memory
        #  Also, need to think how to assign home municipalities better to cover more.
        for thres in [1, 5, 10, 24, 50, 80]:
            # if a user's activity is less than thres, ignore it.
            # but to keep results for different threshold, add thres as a key to the OD.
            if total_presence < thres:
                continue
            munic_flow_count[(home_munic_u, current_munic_u, thres)] += count  # (src, dst, thres) add activity count
            # update unique_user for (src, dst, thres)
            munic_flow_unique_user[(home_munic_u, current_munic_u, thres)].add(uid)

# saving munic level od flow features
munic_flow_count = pd.Series(munic_flow_count).reset_index()
munic_flow_count.columns = ['src', 'dst', 'thres', 'activity']
munic_flow_unique_user = pd.Series(munic_flow_unique_user).reset_index()
munic_flow_unique_user[0] = munic_flow_unique_user[0].apply(len)
munic_flow_unique_user.columns = ['src', 'dst', 'thres', 'n_user']
munic_od = munic_flow_count.merge(munic_flow_unique_user)
logging.info(f'n_munic in src: {munic_od.src.nunique()}, '
             f'n_munic in dst: {munic_od.dst.nunique()}')
munic_od.to_csv(f'data/Study2_municipality_OD_mat-{version}{debug_str}.csv')

print('done')
logging.info('---------------------------------')
