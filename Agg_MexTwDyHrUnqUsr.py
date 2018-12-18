# coding: utf-8
# In[2]:


import datetime
import glob
import gzip
import json
import logging
import os
from collections import defaultdict

from src.creds import mex_root, mex_col_call


# In[3]:


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


# In[5]:


import argparse

parser = argparse.ArgumentParser(
    description='options for aggregating mexico tower daily hourly unique user in call-out or call-in data')
parser.add_argument('--debugging', action='store_true')
parser.add_argument('--call-in-or-out', required=True, choices=['in', 'out'])
args = parser.parse_args()
print(args)

# In[7]:


idx_date = mex_col_call.index('date')
idx_time = mex_col_call.index('time')
idx_t1 = mex_col_call.index('cell_initial')
idx_t2 = mex_col_call.index('cell_final')
idx_duration = mex_col_call.index('duration')
idx_p1 = mex_col_call.index('phone_1')

# In[8]:


mex_root = mex_root
file_pattern = '??????/*TRAFICO_VOZ_[0-9]*' if args.call_in_or_out == 'out' else '??????/*TRAFICO_VOZ_ENTRANTE*'
fns = []
for fn in glob.glob(mex_root + file_pattern):
    fns.append(fn)
fns = sorted(fns)
print(f'nubmer of files to process: {len(fns)}')

# In[9]:


call_in_out_str = 'VOZ' if args.call_in_or_out == 'out' else 'VOZENTRANTE'
stats_dir = f'stats/AggMexTwDyHrUnqUsr{call_in_out_str}/'
if args.debugging: stats_dir += 'debug/'
os.makedirs(stats_dir, exist_ok=True)
print(f'stats_dir = {stats_dir}')

# In[]:

level = logging.DEBUG if args.debugging else logging.INFO
logging.basicConfig(filename=f"logs/AggMexTwDyHrUnqUsr{call_in_out_str}.log", level=level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# In[10]:


done_file_date = set([os.path.basename(fn).replace('.json.gz', '') for fn in glob.glob(stats_dir + '*.json.gz')])
print(f'number of done files on disk: {len(done_file_date)}')

# In[11]:


if os.path.exists(f'{stats_dir}/dates_in_file.json'):
    dates_in_file = json.load(open(f'{stats_dir}/dates_in_file.json'))
    for k, v in dates_in_file.items():
        dates_in_file[k] = set(v)
    dates_in_file = defaultdict(set, dates_in_file)
else:
    dates_in_file = defaultdict(set)

# In[14]:


# loop over the files
start_dt = datetime.datetime.now()
logging.info('===============================')
logging.info(f'MEX stats starts. Number of files: {len(fns)}, '
             f'debugging={args.debugging}, '
             f'call in or out={args.call_in_or_out}')

for cnt, fn in enumerate(fns):

    file_date = fn.replace('.dat', '').replace('.gz', '')[-8:]
    file_date = datetime.datetime.strptime(file_date, '%Y%m%d')
    file_date = file_date.strftime('%Y-%m-%d')
    if file_date in done_file_date:
        print('skipping %dth, file_date: %s' % (cnt, file_date))
        continue

    print('working on the %dth file' % cnt, fn.replace(mex_root, ''))

    logging.info('processing file: %s' % fn.replace(mex_root, ''))

    # for logging processing time per file
    fn_start_dt = datetime.datetime.now()

    # storing stats: stats[d][tower][hour] = set of users
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    # loop over the lines in a file
    # some files are not gzipped
    try:
        f = gzip.open(fn, 'rb') if fn.endswith('.gz') else open(fn, 'rb')

        for i, line in enumerate(f):
            if i > 10 and args.debugging:
                break
            try:
                line = line.decode('utf8').strip().split('|')
                duration = int(line[idx_duration])

                # if duration of a call is larger than 2 hours, discard this call.
                # this will be about 0.02% ~ 0.04% of the total calls
                if duration > 7200:
                    continue

                # get datetime of start and end
                dt1 = datetime.datetime.strptime(line[idx_date] + ' ' + line[idx_time], '%d/%m/%Y %H:%M:%S')
                dt2 = dt1 + datetime.timedelta(seconds=duration)
                # get the date and hour of the datetime
                d1 = dt1.strftime('%Y-%m-%d')
                d1h = dt1.hour
                d2 = dt2.strftime('%Y-%m-%d')
                d2h = dt2.hour
                # update the unique users set
                stats[d1][line[idx_t1]][d1h].add(line[idx_p1])
                stats[d2][line[idx_t2]][d2h].add(line[idx_p1])
                # update the dates_in_files
                dates_in_file[d1].add(file_date)
                dates_in_file[d2].add(file_date)
            except Exception as e:
                if args.debugging: print('file %s line %d raise %s\nThis line is: %s' % (fn, i, type(e).__name__, line))
                logging.exception('file %s line %d raise %s\nThis line is: %s' % (fn, i, type(e).__name__, line))
        f.close()
        logging.debug('iterated all lines')
    except EOFError as e:
        if args.debugging: print(e)
        logging.exception('file %s raise EOFError' % fn)

    # save file
    with gzip.open(stats_dir + '%s.json.gz' % file_date, 'wt') as zipfile:
        json.dump(stats, zipfile, cls=SetEncoder)
    with open(f'{stats_dir}/dates_in_file.json', 'w') as fw:
        json.dump(dates_in_file, fw, cls=SetEncoder)

    logging.info('File processing time: %f seconds' % (datetime.datetime.now() - fn_start_dt).total_seconds())

    if args.debugging:
        break  # fn loop

logging.info('MEX stats ends, %f seconds' % (datetime.datetime.now() - start_dt).total_seconds())
