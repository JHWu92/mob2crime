# coding: utf-8

import pandas as pd
import json
import gzip
import numpy as np
import geopandas as gp
import sys
from collections import defaultdict
import glob
import datetime
import gc

import logging
import src.mex.tower as mex_tower
import src.mex.regions2010 as mex_region

logging.basicConfig(filename="logs/Study2_NewMobFeature.log", level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.info('===============================')
logging.info('Starting Study2_NewMobFeatures v1. ')
logging.info('skip individual mob_ftr bc it is done. This time for munic_flow only')
print('skip individual mob_ftr bc it is done. This time for munic_flow only')
debug = False
logging.info(f'Debug: {debug}')
debug_str = '-debug' if debug else ''


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


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

# In[6]:


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

# In[9]:


dates = sorted([fn.split('/')[-1][:-8] for fn in fns])
if debug:
    dates = dates[:10]


# # count user's presents in municipalities and towers

# In[10]:


def add(d, k1, k2):
    if k1 not in d:
        d[k1] = {}

    if k2 not in d[k1]:
        d[k1][k2] = 0

    d[k1][k2] += 1


# ## count users with night time activities first
# 
# filter users with home activities first
# filter users with home activities above a threshold: at least 1 call every two days at night
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


# In[17]:


# Get user home municipalities
# [user][munic_id]=count
night_mun_count = {}

start = datetime.datetime.now()
for date in dates:
    path = f'stats/AggMexTwDyHrUnqUsrVOZ/{date}.json.gz'
    data = json.load(gzip.open(path))
    for subdate, agg in data.items():
        update_single_count(agg, night_mun_count, kind='home')
    time_spent = (datetime.datetime.now() - start).total_seconds()
    print(date, time_spent, len(night_mun_count))
end = datetime.datetime.now()
time_spent = (end - start).total_seconds()
logging.info(f'Total time spend on night_mun_count {time_spent} seconds')
print(f'Total time spend on night_mun_count {time_spent} seconds')
# In[18]:


uhome = {uid: max(stats, key=stats.get) for uid, stats in night_mun_count.items()}
uhome = pd.Series(uhome).reset_index()
uhome.columns = ['uid', 'home_munic']

# In[19]:


del night_mun_count

gc.collect()

# In[21]:


# Get user activity count at tower level
# [user][gtid]=count
activity_count = {}

start = datetime.datetime.now()
for date in dates:
    path = f'stats/AggMexTwDyHrUnqUsrVOZ/{date}.json.gz'
    data = json.load(gzip.open(path))
    for subdate, agg in data.items():
        update_single_count(agg, activity_count, kind='gtid', uid_set=set(uhome.uid.tolist()))
    print(date, (datetime.datetime.now() - start).total_seconds(), len(activity_count))
end = datetime.datetime.now()
time_spent = (end - start).total_seconds()
logging.info(f'Total time spend on activity_count {time_spent} seconds')
print(f'Total time spend on activity_count {time_spent} seconds')

# In[23]:


uhome['total_presence'] = uhome.uid.apply(lambda x: sum(activity_count[x].values()) if x in activity_count else 0)
logging.info(f'n_user before thres: {len(uhome)}')
thres = len(dates) / 2  # Pappalardo et al 2016 once every two days on average
uhome = uhome[uhome.total_presence > thres]
uhome.set_index('uid', inplace=True)
logging.info(f'thres={thres}, n_user after: {len(uhome)}')
print(f'thres={thres}, n_user after: {len(uhome)}')


# # compute features

# TODO:
#  Maybe, RoG and entropy can be computed at daytime vs night time

# In[25]:


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


# In[73]:


# print('start on mob features')
# start = datetime.datetime.now()
# user_mob_feature = []
# for i, uid in enumerate(uhome.index):
#     if (i + 1) % 5e+4 == 0:
#         print('working on mob features', i)
#         if debug: break
#     if not uid in activity_count:
#         continue
#     rog, ent, cent = get_mob_ftr(activity_count[uid])
#     user_mob_feature.append({'uid': uid, 'RoG': rog, 'Entropy': ent, 'CorrectEntropy': cent})
# user_mob_feature = pd.DataFrame(user_mob_feature)
# user_mob_feature.set_index('uid', inplace=True)
# end = datetime.datetime.now()
# time_spent = (end - start).total_seconds()
# logging.info(f'Total time spend on get_mob_ftr {time_spent} seconds')
# print(f'Total time spend on get_mob_ftr {time_spent} seconds')

# In[ ]:


start = datetime.datetime.now()
user_home_vs_nhome = []
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

print_freq = 1e+3 if debug else 1e+5
print('start on user_home_vs_nhome')
for i, uid in enumerate(uhome.index):
    if (i + 1) % print_freq == 0:
        print('working on home vs non-home presence', i)
        if debug: break
    if not uid in activity_count:
        continue
    home_munic_count = 0
    non_home_count = 0
    home_munic_u = uhome.loc[uid].home_munic
    for gtid, count in activity_count[uid].items():
        # user *uid* appear at tower *gtid* *count* times
        current_munic_u = gt2munic[gtid]
        # it's ok if current==home, it is self-loop.
        munic_flow_count[(home_munic_u, current_munic_u)] += count  # (src, dst) add activity count
        munic_flow_unique_user[(home_munic_u, current_munic_u)].add(uid)  # update unique_user for (src, dst)
        if current_munic_u == home_munic_u:
            home_munic_count += count
        else:
            non_home_count += count
    user_home_vs_nhome.append({'uid': uid, 'home_presence': home_munic_count, 'non_home_presence': non_home_count})

# user_home_vs_nhome = pd.DataFrame(user_home_vs_nhome).set_index('uid')

end = datetime.datetime.now()
time_spent = (end - start).total_seconds()
logging.info(f'Total time spend on user_home_vs_nhome and munic_flow {time_spent} seconds')
print(f'Total time spend on user_home_vs_nhome and munic_flow {time_spent} seconds')
# saving individual features
# uhome.join(user_home_vs_nhome).join(user_mob_feature).to_csv(f'data/Study2_individual_features{debug_str}.csv.gz')

# saving munic level od flow features
munic_flow_count = pd.Series(munic_flow_count).reset_index()
munic_flow_count.columns = ['src', 'dst', 'activity']
munic_flow_unique_user = pd.Series(munic_flow_unique_user).reset_index()
munic_flow_unique_user[0] = munic_flow_unique_user[0].apply(len)
munic_flow_unique_user.columns = ['src', 'dst', 'n_user']
munic_od = munic_flow_count.merge(munic_flow_unique_user)
munic_od.to_csv(f'data/Study2_municipality_OD_mat{debug_str}.csv')
# don't count self-loop
munic_od_ftr = munic_od[munic_od.src != munic_od.dst].groupby('src').agg(
    {'dst': 'count', 'activity': ['sum', 'std'], 'n_user': ['sum', 'std']})
munic_od_ftr.columns = ['_'.join(col).strip() for col in munic_od_ftr.columns.values]
munic_od_ftr.to_csv(f'data/Study2_municipality_flow_ftr{debug_str}.csv')

print('done')
