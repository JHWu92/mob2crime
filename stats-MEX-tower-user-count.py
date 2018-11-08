# coding: utf-8
import datetime
import glob
import gzip
import json
import logging
import os
from collections import defaultdict

from src.creds import drive_root, mex_col_call


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


debugging = False
voz_only = True

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
file_pattern = '??????/*TRAFICO_VOZ_[0-9]*' if voz_only else '??????/*TRAFICO_VOZ_ENTRANTE*'
fns = []
for fn in glob.glob(mex_root + file_pattern):
    fns.append(fn)
fns = sorted(fns)

stats_dir = 'stats/MexTwUniqUser-VOZ/' if voz_only else 'stats/MexTwUniqUser-VOZENTRANTE/'
if debugging: stats_dir += 'debug/'
os.makedirs(stats_dir, exist_ok=True)

done_file_date = set([os.path.basename(fn).replace('.json.gz','') for fn in glob.glob(stats_dir+'*.json.gz')])

# loop over the files
start_dt = datetime.datetime.now()
logging.info('===============================')
logging.info('MEX stats starts. Number of files: %d, debugging=%s, VOZ_only=%s' % (len(fns), debugging, voz_only))

for cnt, fn in enumerate(fns):

    file_date = fn.replace('.dat', '').replace('.gz', '')[-8:]
    file_date = datetime.datetime.strptime(file_date, '%Y%m%d')
    file_date = file_date.strftime('%Y-%m-%d')
    if file_date in done_file_date:
        print('skipping %dth, file_date: %s' % (cnt, file_date))
        continue

    print('working on the %dth file' % cnt)

    logging.info('processing file: %s' % fn.replace(drive_root, ''))

    # for logging processing time per file
    fn_start_dt = datetime.datetime.now()

    # storing stats: stats[d][tower][hour] = set of users
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    # loop over the lines in a file
    # some files are not gzipped
    try:
        f = gzip.open(fn, 'rb') if fn.endswith('.gz') else open(fn, 'rb')

        for i, line in enumerate(f):
            if i > 10 and debugging:
                break

            line = line.decode('utf8').strip().split('|')

            dt1 = datetime.datetime.strptime(line[idx_date] + ' ' + line[idx_time], '%d/%m/%Y %H:%M:%S')
            dt2 = dt1 + datetime.timedelta(seconds=int(line[idx_duration]))

            d1 = dt1.strftime('%Y-%m-%d')
            d1h = dt1.hour
            d2 = dt2.strftime('%Y-%m-%d')
            d2h = dt2.hour

            stats[d1][line[idx_t1]][d1h].add(line[idx_p1])
            stats[d2][line[idx_t2]][d2h].add(line[idx_p1])
        logging.debug('iterated all lines')
        f.close()
    except EOFError as e:
        logging.exception('file %s raise EOFError' % fn)

    # save file
    with gzip.open(stats_dir + '%s.json.gz' % file_date, 'wt') as zipfile:
        json.dump(stats, zipfile, cls=SetEncoder)

    logging.info('File processing time: %f seconds' % (datetime.datetime.now() - fn_start_dt).total_seconds())

    if debugging:
        break  # fn loop

logging.info('MEX stats ends, %f seconds' % (datetime.datetime.now() - start_dt).total_seconds())
