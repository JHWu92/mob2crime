# coding: utf-8

import datetime
import glob
import gzip
import logging
import os
from collections import defaultdict

import pandas as pd

from src.creds import drive_root, mex_col_call

debugging = False
include_in_calls = False

level = logging.DEBUG if debugging else logging.INFO
logging.basicConfig(filename="logs/MEX-tower-user-count.log", level=level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

idx_date = mex_col_call.index('date')
idx_time = mex_col_call.index('time')
idx_t1 = mex_col_call.index('cell_initial')
idx_t2 = mex_col_call.index('cell_final')
idx_duration = mex_col_call.index('duration')
idx_p1 = mex_col_call.index('phone_1')

mex_root = drive_root + "MEX/"
file_pattern = '??????/*TRAFICO_VOZ*' if include_in_calls else '??????/*TRAFICO_VOZ_[0-9]*'
stats_dir = 'stats/MEX-tower-user-count-w-in-calls/' if include_in_calls else 'stats/MEX-tower-user-count/'
os.makedirs(stats_dir, exist_ok=True)

fns = []
for fn in glob.glob(mex_root + file_pattern):
    fns.append(fn)

# loop over the files
start_dt = datetime.datetime.now()
logging.info('===============================')
logging.info('MEX stats starts. debugging=%s, including incoming calls=%s' % (debugging, include_in_calls))

for cnt, fn in enumerate(fns):
    if cnt % 30 ==0:
        print('working on the %dth file' % cnt)

    logging.info('processing file: %s' % fn.replace(drive_root, ''))
    file_date = fn.replace('.dat','').replace('.gz','')[-8:]
    file_date = datetime.datetime.strptime(file_date, '%Y%m%d')
    file_date = file_date.strftime('%d-%m-%Y')

    # for logging processing time per file
    fn_start_dt = datetime.datetime.now()

    # storing stats: stats[d][tower][hour] = set of users
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    # loop over the lines in a file
    # some files are not gzipped
    f = gzip.open(fn, 'rb') if fn.endswith('.gz') else open(fn, 'rb')

    for i, line in enumerate(f):
        if i > 10 and debugging:
            break

        line = line.decode('utf8').strip().split('|')

        dt1 = datetime.datetime.strptime(line[idx_date] + ' ' + line[idx_time], '%d/%m/%Y %H:%M:%S')
        dt2 = dt1 + datetime.timedelta(seconds=int(line[idx_duration]))

        d1 = dt1.strftime('%d-%m-%Y')
        d1h = dt1.hour
        d2 = dt2.strftime('%d-%m-%Y')
        d2h = dt2.hour

        stats[d1][line[idx_t1]][d1h].add(line[idx_p1])
        stats[d2][line[idx_t2]][d2h].add(line[idx_p1])

    logging.debug('iterated all lines')
    f.close()

    # saving the stats
    if file_date in stats:
        logging.debug('file_date in the stats, saving csv')
        today = pd.DataFrame.from_dict(stats.pop(file_date)).applymap(lambda x: pd.np.nan if pd.isnull(x) else len(x))
        today.reindex(list(range(24))).to_csv(stats_dir + '%s.csv' % file_date)
    else:
        logging.warning('file_date not in the stats')

    # there are more dates in addition to file_date
    if len(stats) != 0:
        extra_dates = list(stats.keys())
        logging.warning('There are dates in addition to the file_date: %s' % extra_dates)
        for ed in extra_dates:
            extra = pd.DataFrame.from_dict(stats[ed]).applymap(lambda x: pd.np.nan if pd.isnull(x) else len(x))
            extra.reindex(list(range(24))).to_csv(stats_dir + '%s-from-%s.csv' % (ed, file_date))

    logging.info('File processing time: %f seconds' % (datetime.datetime.now() - fn_start_dt).total_seconds())

    if debugging:
        break  # fn loop

logging.info('MEX stats ends, %f seconds' % (datetime.datetime.now() - start_dt).total_seconds())
