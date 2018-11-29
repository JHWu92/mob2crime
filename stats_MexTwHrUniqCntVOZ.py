import glob
import gzip
import json
import logging
import os
from collections import defaultdict

import pandas as pd

from src.creds import mex_root, mex_tower_08, mex_tower_12

voz_only = True
debugging = False
tower_info_version = '08'

level = logging.DEBUG if debugging else logging.INFO
logging.basicConfig(filename="logs/MexTwHrUniqCnt.log", level=level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stat_dir = 'stats/MexTw%sHrUniqCntVOZ/' % tower_info_version
if debugging:
    stat_dir += 'debug/'
os.makedirs(stat_dir, exist_ok=True)

logging.info('===============================')
logging.info('MEX tower hourly unique user counting starts. debugging=%s, VOZ_only=%s, tower version=%s' % (
    debugging, voz_only, tower_info_version))

# tower ids information
tower_info_path = {'08': mex_root + mex_tower_08, '12': mex_root + mex_tower_12}[tower_info_version]
towers = pd.read_csv(tower_info_path, header=None, sep='|')
towers['latlon'] = towers.apply(lambda x: '%.6f' % (x[2]) + ',' + '%.6f' % (x[3]), axis=1)
towers_gp = towers.groupby('latlon')[0].apply(list).to_frame()
towers_gp['gtid'] = towers_gp[0].apply(lambda x: '-'.join(x))

gt2loc = {row['gtid']: loc.split(',') for loc, row in towers_gp.iterrows()}
t2gt = {}
for _, row in towers_gp.iterrows():
    for tid in row[0]:
        t2gt[tid] = row['gtid']


def update_stat(data, file_date, stats, stats_no_gtid, dates_in_file):
    for date, tw_hr_stats in data.items():
        logging.debug('working of date: %s in file: %s' % (date, file_date))
        # update stats
        for tw, hr_stats in tw_hr_stats.items():
            tid = tw.replace('33F430', '')
            if tid in t2gt:
                gtid = t2gt[tid]
                for hr, uniq_users in hr_stats.items():
                    stats[date][gtid][hr].update(uniq_users)
            else:
                for hr, uniq_users in hr_stats.items():
                    stats_no_gtid[date][tid][hr].update(uniq_users)

        # if files for that date are looped, save the stats and clear the memory
        files = dates_in_file[date]
        files.remove(file_date)
        if len(files) == 0:
            logging.info('saving stats of date %s' % date)
            fn_out = stat_dir + '%s-located.csv' % date
            assert not os.path.exists(fn_out), fn_out
            dstat = stats.pop(date)
            #             dstat = stats[date]
            df = pd.DataFrame.from_dict(dstat).applymap(lambda x: len(x) if not pd.isnull(x) else x)
            df.index = df.index.astype(int)
            df.sort_index().T.to_csv(fn_out)

            fn_out = stat_dir + '%s-no-info.csv' % date
            dstat = stats_no_gtid.pop(date)
            df = pd.DataFrame.from_dict(dstat).applymap(lambda x: len(x) if not pd.isnull(x) else x)
            df.index = df.index.astype(int)
            df.sort_index().T.to_csv(fn_out)
            logging.info('%d towers not in version %s tower info on %s' % (len(dstat), tower_info_version, date))


# storing stats: stats[d][tower][hour] = set of users
stats = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
stats_no_gtid = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
dates_in_file = json.load(open('stats/AggMexTwUniqUserVOZ/dates_in_file.json'))

fns = []
for fn in glob.glob('stats/AggMexTwUniqUserVOZ/*.json.gz'):
    fns.append(fn)
fns = sorted(fns)

for fn in fns:
    print('working on', fn)
    logging.info(fn + ' loading data')
    file_date = os.path.basename(fn).replace('.json.gz', '')
    with gzip.open(fn) as fin:
        data = json.load(fin)
        logging.debug('loaded data')
        update_stat(data, file_date, stats, stats_no_gtid, dates_in_file)
    if debugging:
        break
