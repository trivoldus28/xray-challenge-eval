
from database import Database

def get_top_entries(scores, n=5, metric='erl_norm'):
    scores = sorted(scores, key=lambda x: x[3][metric], reverse=True)
    recorded = set()
    ret = []
    for entry in scores:
        entry_key = (entry[0], entry[1])
        if entry_key in recorded:
            continue
        recorded.add(entry_key)
        ret.append((entry[0], entry[1], entry[2]))
        if len(ret) == n:
            return ret
    return ret

def print_csv(entries, scores):
    print("setup,iteration,threshold,erl_norm,xpress_voi,xpress_rand,xpress_erl_voi,xpress_erl_rand")
    ret = []
    for e in scores:
        # print(e)
        if (e[0], e[1], e[2]) in entries:
            a = [e[0], e[1], e[2], e[3]['erl_norm'], e[3]['xpress_voi'], e[3]['xpress_rand'], e[3]['xpress_erl_voi'], e[3]['xpress_erl_rand']]
            a = [str(k) for k in a]
            a = ','.join(a)
            ret.append(a)
    ret = sorted(ret)
    for a in ret:
        print(a)

if __name__ == '__main__':

    db_name = 'validation_results'
    db = Database(db_name)
    all_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    all_checkpoints = [1000000, 420000, 440000, 460000, 480000, 960000, 970000, 980000, 990000]

    # # get the top 3 thresholds across different metrics
    # checkpoints = [1000000, 960000, 970000, 980000, 990000]
    # scores = db.get_scores(networks=['setup03'], thresholds=all_thresholds, checkpoints=checkpoints)
    # top_entries = set()
    # top_entries |= set(get_top_entries(scores, metric='erl_norm'))
    # top_entries |= set(get_top_entries(scores, metric='xpress_voi'))
    # top_entries |= set(get_top_entries(scores, metric='xpress_rand'))
    # top_entries |= set(get_top_entries(scores, metric='xpress_erl_voi'))
    # top_entries |= set(get_top_entries(scores, metric='xpress_erl_rand'))
    # print(top_entries)
    # print_csv(top_entries, scores)

    # Get scores across all checkpoints
    scores = db.get_scores(networks=['setup03'], thresholds=all_thresholds, checkpoints=None)
    top_entries = set()
    top_entries |= set(get_top_entries(scores, n=16, metric='erl_norm'))
    top_entries |= set(get_top_entries(scores, n=16, metric='xpress_voi'))
    top_entries |= set(get_top_entries(scores, n=16, metric='xpress_rand'))
    top_entries |= set(get_top_entries(scores, n=16, metric='xpress_erl_voi'))
    top_entries |= set(get_top_entries(scores, n=16, metric='xpress_erl_rand'))
    top_entries = sorted(top_entries, key=lambda x: x[1])
    top_entries = sorted(top_entries)
    print(top_entries)
    print_csv(top_entries, scores)


