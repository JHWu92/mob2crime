import gzip
import src.creds as const
import datetime


def day_from_dt(dt):
    return dt.strftime('%Y-%m-%d')


def hour_from_dt(dt):
    return dt.hour


def temporal_info(line, info_prefix='', debug=False, logger=None):
    """

    :param line: list. A record of a call
    :param info_prefix: str, ''. Prefix for printing or logging.
    :param debug: default False. If True, print exception info on terminal
    :param logger: default None. If a logger is passed, log exception info to log files
    :return: duration, dt1(start), dt2(end)
    """
    try:
        duration = int(line[const.idx_duration])
        # if duration of a call is larger than 2 hours, discard this call.
        # this will be about 0.02% ~ 0.04% of the total calls
        if duration > 7200:
            return False

        # get datetime of start(dt1) and end(dt2)
        dt1 = datetime.datetime.strptime(line[const.idx_date] + ' ' + line[const.idx_time], '%d/%m/%Y %H:%M:%S')
        dt2 = dt1 + datetime.timedelta(seconds=duration)
        return duration, dt1, dt2

    except Exception as e:
        info = f'{info_prefix} raise {type(e).__name__}\nThis line is: {line}'
        if debug: print(info)
        if logger is not None: logger.exception(info)
        return False


def update_unique_users_per_line(unique_users, line, info_prefix='', debug=False, logger=None):
    """

    :param unique_users: defaultdict with 3 levels of keys. stats[date][tower][hour] = set of unique phone number
    :param line: list. A record of a call
    :param info_prefix: str, ''. Prefix for printing or logging.
    :param debug: default False. If True, print exception info on terminal
    :param logger: default None. If a logger is passed, log exception info to log files
    :return: duration, dt1(start), dt2(end)
    :return:
    """
    t_info = temporal_info(line, info_prefix, debug, logger)

    # an error is raise in temporal_info() or duration is > 7200
    if t_info is False:
        pass

    duration, dt1, dt2 = t_info
    d1 = day_from_dt(dt1)
    d1h = hour_from_dt(dt1)
    d2 = day_from_dt(dt2)
    d2h = hour_from_dt(dt2)
    t1 = line[const.idx_t1]
    t2 = line[const.idx_t2]
    phone1 = line[const.idx_p1]

    # update the unique users set
    unique_users[d1][t1][d1h].add(phone1)
    unique_users[d2][t2][d2h].add(phone1)


def update_unique_users_by_one_file(fn, unique_users, percentage=None, debug=False, logger=None):
    # loop over the lines in a file
    # some files are not gzipped
    f = gzip.open(fn, 'rb') if fn.endswith('.gz') else open(fn, 'rb')

    for i, line in enumerate(f):
        if i > 10 and debug:
            break
        line = line.decode('utf8').strip().split('|')
        info_prefix = f'file "{fn}" line {i}'
        update_unique_users_per_line(unique_users, line, info_prefix, debug, logger)

    f.close()


def count_and_save_nunique_user_per_day(daily_unique_users):
    return

def clear_done_day(unique_users,day):
    return

