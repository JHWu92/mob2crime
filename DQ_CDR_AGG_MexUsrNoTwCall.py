# coding: utf-8

import datetime
import glob
import gzip
import json
import logging
import os
from collections import defaultdict

from src.creds import mex_root, mex_col_call

# In[4]:


debugging = False
voz_only = False
redo = False

level = logging.DEBUG if debugging else logging.INFO
logging.basicConfig(filename="logs/DQ_CDR_AGG_MexUsrNoTwCall.log", level=level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# In[5]:

idx_t1 = mex_col_call.index('cell_initial')
idx_t2 = mex_col_call.index('cell_final')
idx_p1 = mex_col_call.index('phone_1')

# In[6]:


file_pattern = '??????/*TRAFICO_VOZ_[0-9]*' if voz_only else '??????/*TRAFICO_VOZ_ENTRANTE*'
fns = []
for fn in glob.glob(mex_root + file_pattern):
    fns.append(fn)
fns = sorted(fns)

# In[8]:
stats_dir = 'stats/DQAggMexUsrNoTwCallVOZ/' if voz_only else 'stats/DQAggMexUsrNoTwCallVOZENTRANTE/'
if debugging: stats_dir += 'debug/'
os.makedirs(stats_dir, exist_ok=True)

# In[19]:
if not redo:
    done_file_date = set([os.path.basename(fn).replace('.json.gz', '') for fn in glob.glob(stats_dir + '*.json.gz')])
else:
    done_file_date = set()
    print("REDO is ON")

# In[20]:
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

    print('working on the %dth file' % cnt, fn)

    logging.info('processing file: %s' % fn.replace(mex_root, ''))

    # for logging processing time per file
    fn_start_dt = datetime.datetime.now()

    # storing stats: stats[phone1][kind] = number of call
    stats = defaultdict(lambda: defaultdict(int))

    # loop over the lines in a file
    # some files are not gzipped
    try:
        f = gzip.open(fn, 'rb') if fn.endswith('.gz') else open(fn, 'rb')

        for i, line in enumerate(f):
            if i > 10 and debugging:
                break
            logging.debug(line)
            try:
                line = line.decode('utf8').strip().split('|')
                p1 = line[idx_p1]
                t1 = line[idx_t1]
                t2 = line[idx_t2]
                no_t1 = t1 == '' or t1 is None
                no_t2 = t2 == '' or t2 is None
                stats[p1]['all'] += 1
                if no_t1:
                    stats[p1]['nt1'] += 1
                if no_t2:
                    stats[p1]['nt2'] += 1
                if no_t1 and no_t2:
                    stats[p1]['nt12'] += 1
            except Exception as e:
                logging.exception('file %s line %d raise %s\nThis line is: %s' % (fn, i, type(e).__name__, line))

        logging.debug('iterated all lines')
        f.close()
    except EOFError as e:
        logging.exception('file %s raise EOFError' % fn)

    # save file
    with gzip.open(stats_dir + '%s.json.gz' % file_date, 'wt') as zipfile:
        json.dump(stats, zipfile)

    logging.info('File processing time: %f seconds' % (datetime.datetime.now() - fn_start_dt).total_seconds())

    if debugging and cnt >= 1:
        break  # fn loop

logging.info('MEX stats ends, %f seconds' % (datetime.datetime.now() - start_dt).total_seconds())
