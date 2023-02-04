
import sqlite3
import json

class Database():

    def __init__(self, db_name, table_name="scores_table"):
        self.conn = sqlite3.connect(f'{db_name}.db', check_same_thread=False)
        self.table_name = table_name

        self.cursor = self.conn.cursor()
        # check if table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [k[0] for k in self.cursor.fetchall()]
        if self.table_name not in tables:
            self.cursor.execute(f"CREATE TABLE {self.table_name} (network text, checkpoint int, threshold real, scores text)")
            self.conn.commit()

    def add_score(self, network, checkpoint, threshold, scores_dict):
        assert type(network) is str
        assert type(checkpoint) is int
        assert type(threshold) is float
        assert type(scores_dict) is dict
        scores_str = json.dumps(scores_dict)
        self.cursor.execute(f"INSERT INTO {self.table_name} VALUES ('{network}', {checkpoint}, {threshold}, '{scores_str}')")
        self.conn.commit()

    def get_scores(self, networks=None, checkpoints=None, thresholds=None):
        assert type(networks) is str or networks is None or type(networks) is list
        assert type(checkpoints) is int or checkpoints is None or type(checkpoints) is list
        assert type(thresholds) is float or thresholds is None or type(thresholds) is list

        def to_csv_list(l):
            return ','.join([f'\'{ll}\'' for ll in l])

        conditioned = False
        def add_where(var, var_name, query):
            nonlocal conditioned
            ret = ''
            if var is not None:
                if conditioned:
                    ret += " and "
                else:
                    ret += " where "
                conditioned = True
                if type(var) is str:
                    ret +=  f"{var_name} = '{var}'"
                else:
                    ret +=  f"{var_name} in ({to_csv_list(var)})"
            # print(ret)
            return ret

        query = f"SELECT * FROM {self.table_name}"
        query += add_where(networks, "network", query)
        query += add_where(checkpoints, "checkpoint", query)
        query += add_where(thresholds, "threshold", query)
        # print(query)
        ret = self.cursor.execute(query).fetchall()
        # print(ret)
        ret = [list(k) for k in ret]
        for item in ret:
            item[3] = json.loads(item[3])
        # print(ret)
        return ret
