# coding: utf-8

# In[1]:


import argparse
import datetime
import glob
import gzip
import logging
import os

from src.creds import mex_root, mex_col_call

# In[ ]:
parser = argparse.ArgumentParser()
parser.add_argument('--call-in-or-out', required=True, choices=['in', 'out'])
args = parser.parse_args()
print(args)

# In[2]:


call_in_or_out = args.call_in_or_out

# In[7]:


idx_date = mex_col_call.index('date')
idx_time = mex_col_call.index('time')
idx_t1 = mex_col_call.index('cell_initial')
idx_t2 = mex_col_call.index('cell_final')
idx_duration = mex_col_call.index('duration')
idx_p1 = mex_col_call.index('phone_1')

# In[4]:


mex_root = mex_root
file_pattern = '??????/*TRAFICO_VOZ_[0-9]*' if call_in_or_out == 'out' else '??????/*TRAFICO_VOZ_ENTRANTE*'
fns = []
for fn in glob.glob(mex_root + file_pattern):
    fns.append(fn)
fns = sorted(fns)
print(f'nubmer of files to process: {len(fns)}')

# In[9]:


call_in_out_str = 'VOZ' if call_in_or_out == 'out' else 'VOZENTRANTE'
stats_dir = f'stats/DQMexTimeAnomaly{call_in_out_str}/'
os.makedirs(stats_dir, exist_ok=True)
print(f'stats_dir = {stats_dir}')

# In[15]:


# loop over the files
start_dt = datetime.datetime.now()

for cnt, fn in enumerate(fns):

    file_date = fn.replace('.dat', '').replace('.gz', '')[-8:]
    file_date = datetime.datetime.strptime(file_date, '%Y%m%d')
    file_date = file_date.strftime('%Y-%m-%d')

    print('working on the %dth file' % cnt, fn.replace(mex_root, ''))

    # storing stats: stats[d][tower][hour] = set of users
    blines = []

    # loop over the lines in a file
    # some files are not gzipped
    try:
        f = gzip.open(fn, 'rb') if fn.endswith('.gz') else open(fn, 'rb')

        for i, line in enumerate(f):
            # if i > 10:
            #     break
            try:
                bit_line = line
                line = line.decode('utf8').strip().split('|')
                # get datetime of start and end
                dt1 = datetime.datetime.strptime(line[idx_date] + ' ' + line[idx_time], '%d/%m/%Y %H:%M:%S')
                dur = int(line[idx_duration])
                # get the date and hour of the datetime
                d1 = dt1.strftime('%Y-%m-%d')
                # the startdate is not the file date or the duration is larger than 2hours
                if d1 != file_date or dur > 7200:
                    blines.append(bit_line)

            except Exception as e:
                print('file %s line %d raise %s\nThis line is: %s' % (fn, i, type(e).__name__, line))
        f.close()
        logging.debug('iterated all lines')
    except EOFError as e:
        print('file %s raise EOFError' % fn)

    print(f'number of anormal lines = {len(blines)}')
    # save file
    with gzip.open(stats_dir + '%s.dat.gz' % file_date, 'w') as zipfile:
        zipfile.writelines(blines)

print('MEX stats ends, %f seconds' % (datetime.datetime.now() - start_dt).total_seconds())
