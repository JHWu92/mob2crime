from src.features import *


def grid_avgerage(tw_avg, t2g):
    # there are grids without any call throughout the observation period
    g_avg = t2g.merge(tw_avg, left_on='gtid', right_index=True, how='left')

    for h in range(24):
        h = str(h)
        g_avg[h] = g_avg[h] * g_avg['weight']

    g_avg = g_avg.drop(['gtid', 'weight'],
                       axis=1).groupby('grid').sum()  # fillna=0 by default

    return g_avg


aver = mex.stat_tw_dow_aver_hr_uniq_user('out+in')
tw_avg_wd = pd.DataFrame(aver['wd']).T
tw_avg_wk = pd.DataFrame(aver['wk']).T
gside = 500
mex_t2g_mpol = mex.tower2grid('metropolitans_16', gside)
print(
    'number of towers in cities has no call at all during weekday and weekend',
    len(set(mex_t2g_mpol.gtid) - set(tw_avg_wd.index)),
    len(set(mex_t2g_mpol.gtid) - set(tw_avg_wk.index)))
# g_avg = pd.DataFrame([average number of calls], index=grid, columns='hour')
g_avg_wd_mpol = grid_avgerage(tw_avg_wd, mex_t2g_mpol)
g_avg_wk_mpol = grid_avgerage(tw_avg_wk, mex_t2g_mpol)

dv_r_mpol = urban_dilatation_index(g_avg_wd_mpol, 'metropolitans_16', 'metropolitan', gside)
print(dv_r_mpol)
