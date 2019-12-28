# coding: utf-8

# In[1]:

import argparse
import gzip
import json
import logging
import os
from collections import defaultdict, deque

import pandas as pd

from src.creds import mex_root, mex_tower_fn

# In[157]:
# =================
# argument settings
# =================
parser = argparse.ArgumentParser(
    description='options for aggregating mexico tower daily hourly unique user in call-out or call-in data')
parser.add_argument('--debugging', action='store_true')
parser.add_argument('--call-in-or-out', required=True, choices=['in', 'out', 'out+in'])
parser.add_argument('--trange', required=True, choices=['24', '4'])
args = parser.parse_args()
print(args)

HOUR_TO_TIME_RANGE = {
    '24': {i: i for i in range(24)},
    '4': {
        **{h: 0 for h in [23, 0, 1, 2, 3, 4]},  # 11pm to 5am
        **{h: 1 for h in [5, 6, 7, 8, 9, 10]},  # 5am to 11am
        **{h: 2 for h in [11, 12, 13, 14, 15, 16]},  # 11am to 5pm
        **{h: 3 for h in [17, 18, 19, 20, 21, 22]},  # 6pm to 11pm
    }
}
print('HOUR_TO_TIME_RANGE', HOUR_TO_TIME_RANGE[args.trange])

level = logging.DEBUG if args.debugging else logging.INFO
logging.basicConfig(filename="logs/StatMexTwHrUniqCnt.log", level=level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# directory to store the number of user count
stat_dir = f'stats/MexTwHrUniqCnt-{args.call_in_or_out}-TR{args.trange}/'
if args.debugging:
    stat_dir += 'debug/'
print('store statistics in ', stat_dir)
os.makedirs(stat_dir, exist_ok=True)

logging.info('===============================')
logging.info('MEX tower hourly unique user counting starts. '
             f'debugging={args.debugging}, call_in_or_out={args.call_in_or_out}')

# In[13]:
date = ''

# tower ids information
tower_info_path = mex_root + mex_tower_fn
towers = pd.read_csv(tower_info_path, header=None, sep='|')
towers['latlon'] = towers.apply(lambda x: '%.6f' % (x[2]) + ',' + '%.6f' % (x[3]), axis=1)
towers_gp = towers.groupby('latlon')[0].apply(list).to_frame()
towers_gp['gtid'] = towers_gp[0].apply(lambda x: '-'.join(x))

gt2loc = {row['gtid']: loc.split(',') for loc, row in towers_gp.iterrows()}
t2gt = {}
for _, row in towers_gp.iterrows():
    for tid in row[0]:
        t2gt[tid] = row['gtid']


# In[140]:


class DataQueue:
    def __init__(self, directory, maxlen=6):
        self.directory = directory
        self.maxlen = maxlen
        self.dates = deque(maxlen=6)
        self.aggs = {}

    def get_agg(self, date):
        if date not in self.aggs:
            loaded = self.load_agg(date)
            if not loaded:
                #                 print(f'get_agg {stat_date} can not be loaded')
                return None
        return self.aggs[date]

    def load_agg(self, date):
        if set(self.dates) != set(self.aggs.keys()):
            self.dates = deque([d for d in self.dates if d in self.aggs])
        fn = f'{self.directory}{date}.json.gz'
        if not os.path.exists(fn):
            return False
        if len(self.dates) == self.maxlen:
            self.aggs.pop(self.dates.popleft())
        self.dates.append(date)
        agg = json.load(gzip.open(fn))
        #         agg_for_sdate=fn
        self.aggs[date] = agg
        return True


# In[154]:


def update_stat_by_agg(local_stat, local_stat_no_tinfo, agg):
    # compiling stat: stats[tower][hour] = set of users
    for tid, hr_uniq_users in agg.items():
        tid = tid.replace('33F430', '')
        if tid in t2gt:
            gtid = t2gt[tid]
            for hr, uniq_users in hr_uniq_users.items():
                tr_bin = HOUR_TO_TIME_RANGE[args.trange][int(hr)]
                local_stat[gtid][tr_bin].update(uniq_users)
        else:
            for hr, uniq_users in hr_uniq_users.items():
                tr_bin = HOUR_TO_TIME_RANGE[args.trange][int(hr)]
                local_stat_no_tinfo[tid][tr_bin].update(uniq_users)


# In[159]:


call_directions = args.call_in_or_out.split('+')

# In[160]:


# initiating task list
agg_dir = {'in': 'stats/AggMexTwDyHrUnqUsrVOZENTRANTE/', 'out': 'stats/AggMexTwDyHrUnqUsrVOZ/'}
dates_in_file = {}
data_queues = {}
dates = set()
for call_d in call_directions:
    adir = agg_dir[call_d]
    dates_in_file[call_d] = json.load(open(f'{adir}dates_in_file.json'))
    dates.update(dates_in_file[call_d].keys())
    data_queues[call_d] = DataQueue(directory=adir, maxlen=6)

dates = sorted(dates)

# In[176]:


# for each stat_date
for stat_date in dates:
    print(f'working on date: {stat_date}')
    logging.info(f'working on date: {stat_date}')

    # compiling stat: stats[tower][hour] = set of users in one day
    stat = defaultdict(lambda: defaultdict(set))
    stat_no_tinfo = defaultdict(lambda: defaultdict(set))

    # get agg_file_dates in each call direction of interests for each stat_date
    for call_d in call_directions:
        if stat_date not in dates_in_file[call_d]:
            print(f'{stat_date} not in call_direction: {call_d}')
            logging.info(f'{stat_date} not in call_direction: {call_d}')
            continue
        agg_dates = dates_in_file[call_d][stat_date]
        if args.debugging: print(call_d, agg_dates)
        # update the unique users set of each hour in that stat_date by each agg_file in each call direction
        for adate in agg_dates:
            agg_for_sdate = data_queues[call_d].get_agg(adate)[stat_date]
            update_stat_by_agg(stat, stat_no_tinfo, agg_for_sdate)
    #             break
    # store stat
    df_stat = pd.DataFrame(stat).T.applymap(lambda x: len(x) if not pd.isnull(x) else 0)
    df_stat.to_csv(f"{stat_dir}{stat_date}-located.csv")
    df_stat_no_tinfo = pd.DataFrame(stat_no_tinfo).T.applymap(lambda x: len(x) if not pd.isnull(x) else 0)
    df_stat_no_tinfo.to_csv(f"{stat_dir}{stat_date}-no-info.csv")
    logging.info('%d towers located, %d towers no info' % (len(df_stat), len(df_stat_no_tinfo)))
    if args.debugging:
        break

# In[ ]:


logging.info('finish counting')
logging.info('*' * 20)
