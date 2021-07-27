import psycopg2 as pg
import pdb
import time
import random
from query_representation.utils import *

import time
from collections import OrderedDict, defaultdict
# from multiprocessing import Pool
import re
import pandas as pd
import numpy as np
import os

CREATE_TABLE_TEMPLATE = "CREATE TABLE {name} (id SERIAL, {columns})"
INSERT_TEMPLATE = "INSERT INTO {name} ({columns}) VALUES %s"

NTILE_CLAUSE = "ntile({BINS}) OVER (ORDER BY {COLUMN}) AS {ALIAS}"
GROUPBY_TEMPLATE = "SELECT {COLS}, COUNT(*) FROM {FROM_CLAUSE} GROUP BY {COLS}"
COUNT_SIZE_TEMPLATE = "SELECT COUNT(*) FROM {FROM_CLAUSE}"

SELECT_ALL_COL_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL"
ALIAS_FORMAT = "{TABLE} AS {ALIAS}"
MIN_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL ORDER BY {COL} ASC LIMIT 1"
MAX_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL ORDER BY {COL} DESC LIMIT 1"
UNIQUE_VALS_TEMPLATE = "SELECT DISTINCT {COL} FROM {FROM_CLAUSE}"
UNIQUE_COUNT_TEMPLATE = "SELECT COUNT(*) FROM (SELECT DISTINCT {COL} from {FROM_CLAUSE}) AS t"

INDEX_LIST_CMD = """
select
    t.relname as table_name,
    a.attname as column_name,
    i.relname as index_name
from
    pg_class t,
    pg_class i,
    pg_index ix,
    pg_attribute a
where
    t.oid = ix.indrelid
    and i.oid = ix.indexrelid
    and a.attrelid = t.oid
    and a.attnum = ANY(ix.indkey)
    and t.relkind = 'r'
order by
    t.relname,
    i.relname;"""


RANGE_PREDS = ["gt", "gte", "lt", "lte"]

CREATE_INDEX_TMP = '''CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON {TABLE} ({COLUMN});'''

class Featurizer():
    def __init__(self, user, pwd, db_name, db_host ,port):
        '''
        '''
        self.user = user
        self.pwd = pwd
        self.db_host = db_host
        self.port = port
        self.db_name = db_name
        self.ckey = "cardinality"

        # if "LCARD_USER" in os.environ:
            # self.user = os.environ["LCARD_USER"]
        # if "LCARD_PORT" in os.environ:
            # self.port = os.environ["LCARD_PORT"]

        # stats on the used columns
        #   table_name : column_name : attribute : value
        #   e.g., stats["title"]["id"]["max_value"] = 1010112
        #         stats["title"]["id"]["type"] = int
        #         stats["title"]["id"]["num_values"] = x
        self.column_stats = {}

        self.max_discrete_featurizing_buckets = None

        self.continuous_feature_size = 2

        self.featurizer = None
        self.cmp_ops = set()
        self.tables = set()
        self.joins = set()
        self.aliases = {}
        self.cmp_ops_onehot = {}
        self.regex_cols = set()

        self.templates = []

        # for pgm stuff
        # self.foreign_keys = {}    # table.key : table.key
        # self.primary_keys = set() # table.key
        # self.alias_to_keys = defaultdict(set)

        self.max_in_degree = 0
        self.max_out_degree = 0
        self.max_paths = 0
        self.feat_num_paths = False

        # the node-edge connectivities stay constant through templates
        # key: template_name
        # val: {}:
        #   key: node name
        #   val: tolerance in powers of 10
        self.template_info = {}

        # things like tolerances, flows need to be computed on a per query
        # basis (maybe we should not precompute these?)
        self.query_info = {}

        # these need to be consistent across all featurizations for this db
        self.max_tables = 0
        self.max_joins = 0
        self.max_preds = 0

    def update_column_stats(self, qreps):
        for qrep in qreps:
            self._update_stats(qrep)

    def update_ystats(self, qreps):
        y = np.array(get_all_cardinalities(qreps, self.ckey))
        if self.ynormalization == "log":
            y = np.log(y)

        self.max_val = np.max(y)
        self.min_val = np.min(y)

    def execute(self, sql):
        '''
        '''
        # archive only considers the stuff stored in disk
        ## FIXME: get stuff that works on both places
        # works on aws
        # con = pg.connect(user=self.user, port=self.port,
                # password=self.pwd, database=self.db_name)

        con = pg.connect(user=self.user, host=self.db_host, port=self.port,
                password=self.pwd, database=self.db_name)
        cursor = con.cursor()

        try:
            cursor.execute(sql)
        except Exception as e:
            print("query failed to execute: ", sql)
            print(e)
            cursor.execute("ROLLBACK")
            con.commit()
            cursor.close()
            con.close()
            print("returning arbitrary large value for now")
            return [[1000000]]
            # return None

        try:
            exp_output = cursor.fetchall()
        except Exception as e:
            print(e)
            exp_output = None

        cursor.close()
        con.close()

        return exp_output

    def setup(self, heuristic_features=True,
            ynormalization="log",
            table_features = True,
            pred_features = True,
            join_features = True,
            flow_features = False,
            num_tables_feature=False,
            separate_regex_bins=True,
            separate_cont_bins=True,
            featurization_type="combined",
            max_discrete_featurizing_buckets=10,
            feat_num_paths= False, feat_flows=False,
            feat_pg_costs = True, feat_tolerance=True,
            feat_template=False, feat_pg_path=True,
            feat_rel_pg_ests=True, feat_join_graph_neighbors=True,
            feat_rel_pg_ests_onehot=True,
            feat_pg_est_one_hot=True,
            cost_model=None, sample_bitmap=False, sample_bitmap_num=1000,
            sample_bitmap_buckets=1000,
            featkey=None):
        '''
        Sets up a transformation to 1d feature vectors based on the registered
        templates seen in get_samples.
        Generates a featurizer dict:
            {table_name: (idx, num_vals)}
            {join_key: (idx, num_vals)}
            {pred_column: (idx, num_vals)}
        where the idx refers to the elements position in the feature vector,
        and num_vals refers to the number of values it will occupy.
        E.g. TODO.
        '''
        args = locals()
        arg_key = ""
        for k, val in args.items():
            if k == "self":
                continue
            self.__setattr__(k, val)
            arg_key += str(val)
        self.featkey = str(deterministic_hash(arg_key))

        # let's figure out the feature len based on db.stats
        assert self.featurizer is None

        # only need to know the number of tables for table features
        self.table_featurizer = {}
        # sort the tables so the features are reproducible
        for i, table in enumerate(sorted(self.tables)):
            self.table_featurizer[table] = i

        if self.sample_bitmap:
            bitmap_tables = []
            self.sample_bitmap_key = "sb" + str(self.sample_bitmap_num)
            self.bitmap_mapping = {}
            self.bitmap_next_mapping = {}

            for i, table in enumerate(sorted(self.tables)):
                if table in SAMPLE_TABLES:
                    bitmap_tables.append(table)
                    self.bitmap_next_mapping[table] = 0

            bitmap_tables.sort()
            # also indexes into table_featurizer
            self.sample_bitmap_featurizer = {}
            table_idx = len(self.tables)
            self.max_table_feature_len = 0
            for i, table in enumerate(bitmap_tables):
                # how many elements in the current table
                count_str = "SELECT COUNT(*) FROM {}".format(table)
                output = self.execute(count_str)
                count = output[0][0]
                feat_count = min(count, sample_bitmap_buckets)
                self.sample_bitmap_featurizer[table] = (table_idx, feat_count)
                table_idx += feat_count

                cur_table_feature_len = feat_count + len(self.tables)
                if cur_table_feature_len > self.max_table_feature_len:
                    self.max_table_feature_len = cur_table_feature_len

            self.table_features_len = table_idx
        else:
            self.table_features_len = len(self.tables)
            self.max_table_feature_len = len(self.tables)

        self.join_featurizer = {}

        for i, join in enumerate(sorted(self.joins)):
            self.join_featurizer[join] = i

        # self.max_discrete_featurizing_buckets = max_discrete_featurizing_buckets
        self.featurizer = {}
        self.num_cols = len(self.column_stats)
        all_cols = list(self.column_stats.keys())
        all_cols.sort()
        self.columns_onehot_idx = {}
        for cidx, col_name in enumerate(all_cols):
            self.columns_onehot_idx[col_name] = cidx

        self.pred_features_len = 0
        for i, cmp_op in enumerate(sorted(self.cmp_ops)):
            self.cmp_ops_onehot[cmp_op] = i

        # to find the number of features, need to go over every column, and
        # choose how many spots to keep for them
        col_keys = list(self.column_stats.keys())
        col_keys.sort()

        self.max_pred_len = 0

        for col in col_keys:
            info = self.column_stats[col]
            pred_len = 0
            # for operator type
            pred_len += len(self.cmp_ops)

            if self.heuristic_features:
                # for pg_est of current table
                pred_len += 1
                if self.featurization_type == "set":
                    # for pg est of current query
                    pred_len += 1

            if is_float(info["min_value"]) and is_float(info["max_value"]) \
                    and info["num_values"] >= self.max_discrete_featurizing_buckets:
                # then use min-max normalization, no matter what
                # only support range-queries, so lower / and upper predicate
                pred_len += self.continuous_feature_size
                continuous = True
            else:
                # so they don't clash with each other
                if self.separate_cont_bins \
                    and self.featurization_type == "set":
                    pred_len += self.continuous_feature_size

                # use 1-hot encoding
                num_buckets = min(self.max_discrete_featurizing_buckets,
                        info["num_values"])
                pred_len += num_buckets
                continuous = False

                if col in self.regex_cols:
                    # give it more space for #num-chars, #number in regex or
                    # not
                    pred_len += 2
                    if self.separate_regex_bins:
                        # extra space for regex buckets
                        pred_len += num_buckets

            self.featurizer[col] = (self.pred_features_len, pred_len, continuous)
            self.pred_features_len += pred_len

            if self.max_pred_len < pred_len:
                self.max_pred_len = pred_len

        if self.featurization_type == "set":
            print("""adding one-hot vector to specify which column \
                    predicate's column""")
            self.max_pred_len += self.num_cols
            print("maximum length of single pred feature: ", self.max_pred_len)

        # for pg_est of all features combined
        if self.heuristic_features:
            self.pred_features_len += 1

        # for num_tables present
        if self.num_tables_feature:
            self.pred_features_len += 1

        ## flow feature things
        self.PG_EST_BUCKETS = 7
        if self.flow_features:
            # num flow features: concat of 1-hot vectors
            self.num_flow_features = 0
            self.num_flow_features += self.max_in_degree+1
            self.num_flow_features += self.max_out_degree+1

            self.num_flow_features += len(self.aliases)

            # for heuristic estimate for the node
            self.num_flow_features += 1

            # for normalized value of num_paths
            if self.feat_num_paths:
                self.num_flow_features += 1
            if self.feat_pg_costs:
                self.num_flow_features += 1
            if self.feat_tolerance:
                # 1-hot vector based on dividing/multiplying value by 10...10^4
                self.num_flow_features += 4
            if self.feat_flows:
                self.num_flow_features += 1

            if self.feat_template:
                self.num_flow_features += len(self.templates)

            if self.feat_pg_path:
                self.num_flow_features += 1

            if self.feat_rel_pg_ests:
                # current node size est, relative to total cost
                self.num_flow_features += 1

                # current node est, relative to all neighbors in the join graph
                # we will hard code the neighbor into a 1-hot vector
                self.num_flow_features += len(self.table_featurizer)

            if self.feat_rel_pg_ests_onehot:
                self.num_flow_features += self.PG_EST_BUCKETS
                # 2x because it can be smaller or larger
                self.num_flow_features += \
                    2*len(self.table_featurizer)*self.PG_EST_BUCKETS

            if self.feat_join_graph_neighbors:
                self.num_flow_features += len(self.table_featurizer)

            if self.feat_pg_est_one_hot:
                # upto 10^7
                self.num_flow_features += self.PG_EST_BUCKETS

            # pg est for the node
            self.num_flow_features += len(self.cmp_ops)
            self.num_flow_features += 1

    def get_onehot_bucket(self, num_buckets, base, val):
        assert val >= 1.0
        for i in range(num_buckets):
            if val > base**i and val < base**(i+1):
                return i

        return num_buckets

    def get_flow_features(self, node, subsetg,
            template_name, join_graph, cmp_op):
        assert node != SOURCE_NODE
        ckey = "cardinality"
        flow_features = np.zeros(self.num_flow_features, dtype=np.float32)
        cur_idx = 0
        # incoming edges
        in_degree = subsetg.in_degree(node)
        flow_features[cur_idx + in_degree] = 1.0
        cur_idx += self.max_in_degree+1
        # outgoing edges
        out_degree = subsetg.out_degree(node)
        flow_features[cur_idx + out_degree] = 1.0
        cur_idx += self.max_out_degree+1
        # num tables
        max_tables = len(self.aliases)
        nt = len(node)
        # assert nt <= max_tables
        flow_features[cur_idx + nt] = 1.0
        cur_idx += max_tables

        # precomputed based stuff
        if self.feat_num_paths:
            if node in self.template_info[template_name]:
                num_paths = self.template_info[template_name][node]["num_paths"]
            else:
                num_paths = 0

            # assuming min num_paths = 0, min-max normalization
            flow_features[cur_idx] = num_paths / self.max_paths
            cur_idx += 1

        if self.feat_pg_costs and self.heuristic_features:
            in_edges = subsetg.in_edges(node)
            in_cost = 0.0
            for edge in in_edges:
                in_cost += subsetg[edge[0]][edge[1]][self.cost_model + "pg_cost"]
            # normalized pg cost
            flow_features[cur_idx] = in_cost / subsetg.graph[self.cost_model + "total_cost"]
            cur_idx += 1

        if self.feat_tolerance:
            tol = subsetg.nodes()[node]["tolerance"]
            tol_idx = int(np.log10(tol))
            assert tol_idx <= 4
            flow_features[cur_idx + tol_idx-1] = 1.0
            cur_idx += 4

        if self.feat_flows and self.heuristic_features:
            in_edges = subsetg.in_edges(node)
            in_flows = 0.0
            for edge in in_edges:
                in_flows += subsetg[edge[0]][edge[1]]["pg_flow"]
            # normalized pg flow
            flow_features[cur_idx] = in_flows
            cur_idx += 1

        if self.feat_template:
            tidx = 0
            for i,t in enumerate(sorted(self.templates)):
                if t == template_name:
                    tidx = i
            flow_features[cur_idx + tidx] = 1.0
            cur_idx += len(self.templates)

        if self.feat_pg_path:
            if "pg_path" in subsetg.nodes()[node]:
                flow_features[cur_idx] = 1.0

        if self.feat_join_graph_neighbors:
            # neighbors = nx.node_boundary(join_graph, node)
            neighbors = list(nx.node_boundary(join_graph, node))
            neighbors.sort()

            for al in neighbors:
                if al not in self.aliases:
                    # possible, for instance with set featurization - we don't
                    # reserve spots for all possible columns / tables in the
                    # unseen set
                    continue
                table = self.aliases[al]
                tidx = self.table_featurizer[table]
                flow_features[cur_idx + tidx] = 1.0
            cur_idx += len(self.table_featurizer)

        if self.feat_rel_pg_ests and self.heuristic_features:
            total_cost = subsetg.graph[self.cost_model+"total_cost"]
            pg_est = subsetg.nodes()[node][ckey]["expected"]
            flow_features[cur_idx] = pg_est / total_cost
            cur_idx += 1
            neighbors = list(nx.node_boundary(join_graph, node))
            neighbors.sort()

            # neighbors in join graph
            for al in neighbors:
                if al not in self.aliases:
                    continue
                table = self.aliases[al]
                tidx = self.table_featurizer[table]
                ncard = subsetg.nodes()[tuple([al])][ckey]["expected"]
                # TODO: should this be normalized? how?
                flow_features[cur_idx + tidx] = pg_est / ncard
                flow_features[cur_idx + tidx] /= 1e5

            cur_idx += len(self.table_featurizer)

        if self.feat_rel_pg_ests_onehot \
            and self.heuristic_features:
            total_cost = subsetg.graph[self.cost_model+"total_cost"]
            pg_est = subsetg.nodes()[node][ckey]["expected"]
            # flow_features[cur_idx] = pg_est / total_cost
            pg_ratio = total_cost / float(pg_est)

            bucket = self.get_onehot_bucket(self.PG_EST_BUCKETS, 10, pg_ratio)
            flow_features[cur_idx+bucket] = 1.0
            cur_idx += self.PG_EST_BUCKETS

            # neighbors = nx.node_boundary(join_graph, node)
            neighbors = list(nx.node_boundary(join_graph, node))
            neighbors.sort()

            # neighbors in join graph
            for al in neighbors:
                if al not in self.aliases:
                    continue
                table = self.aliases[al]
                tidx = self.table_featurizer[table]
                ncard = subsetg.nodes()[tuple([al])][ckey]["expected"]
                # TODO: should this be normalized? how?
                # flow_features[cur_idx + tidx] = pg_est / ncard
                # flow_features[cur_idx + tidx] /= 1e5
                if pg_est > ncard:
                    # first self.PG_EST_BUCKETS
                    bucket = self.get_onehot_bucket(self.PG_EST_BUCKETS, 10,
                            pg_est / float(ncard))
                    flow_features[cur_idx+bucket] = 1.0
                else:
                    bucket = self.get_onehot_bucket(self.PG_EST_BUCKETS, 10,
                            float(ncard) / pg_est)
                    flow_features[cur_idx+self.PG_EST_BUCKETS+bucket] = 1.0

                cur_idx += 2*self.PG_EST_BUCKETS

        if self.feat_pg_est_one_hot and self.heuristic_features:
            pg_est = subsetg.nodes()[node][ckey]["expected"]

            for i in range(self.PG_EST_BUCKETS):
                if pg_est > 10**i and pg_est < 10**(i+1):
                    flow_features[cur_idx+i] = 1.0
                    break

            if pg_est > 10**self.PG_EST_BUCKETS:
                flow_features[cur_idx+self.PG_EST_BUCKETS] = 1.0
            cur_idx += self.PG_EST_BUCKETS

        # pg_est for node will be added in query_dataset..
        if cmp_op is not None:
            cmp_idx = self.cmp_ops_onehot[cmp_op]
            flow_features[cur_idx + cmp_idx] = 1.0
        cur_idx += len(self.cmp_ops)

        return flow_features

    def get_table_features(self, table, bitmap_dict=None):
        '''
        '''
        if self.featurization_type == "set":
            tables_vector = np.zeros(self.max_table_feature_len)
            if table not in self.table_featurizer:
                # print("table: {} not found in featurizer".format(table))
                return tables_vector
            tables_vector[self.table_featurizer[table]] = 1.00
            if bitmap_dict is not None and self.sample_bitmap:
                if self.sample_bitmap_key not in bitmap_dict:
                    return tables_vector
                bitmap = bitmap_dict[self.sample_bitmap_key]
                _, num_bins = self.sample_bitmap_featurizer[table]

                for val in bitmap:
                    if table+str(val) in self.bitmap_mapping:
                        cur_bin = self.bitmap_mapping[table+str(val)]
                    else:
                        cur_bin = self.bitmap_next_mapping[table]
                        self.bitmap_next_mapping[table] += 1
                        self.bitmap_mapping[table+str(val)] = cur_bin

                    idx = cur_bin % num_bins

                    #### only difference compared to the non-set case. combine
                    #### the code.
                    tables_vector[idx] = 1.00
            return tables_vector

        else:
            tables_vector = np.zeros(self.table_features_len)
            if table not in self.table_featurizer:
                print("table: {} not found in featurizer".format(table))
                return tables_vector

            tables_vector[self.table_featurizer[table]] = 1.00
            if bitmap_dict is not None and self.sample_bitmap:
                if self.sample_bitmap_key not in bitmap_dict:
                    return tables_vector
                bitmap = bitmap_dict[self.sample_bitmap_key]
                start_idx, num_bins = self.sample_bitmap_featurizer[table]
                # print(start_idx, num_bins)
                # pdb.set_trace()

                for val in bitmap:
                    if table+str(val) in self.bitmap_mapping:
                        cur_bin = self.bitmap_mapping[table+str(val)]
                    else:
                        cur_bin = self.bitmap_next_mapping[table]
                        self.bitmap_next_mapping[table] += 1
                        self.bitmap_mapping[table+str(val)] = cur_bin

                    idx = cur_bin % num_bins
                    tables_vector[start_idx+idx] = 1.00

            return tables_vector

    def get_join_features(self, join_str):
        if self.featurization_type == "set":
            # essentially, exactly the same as w/o set, since we just used a
            # 1-hot encoding to represent these
            keys = join_str.split("=")
            keys.sort()
            keys = ",".join(keys)
            joins_vector = np.zeros(len(self.join_featurizer))
            if keys not in self.join_featurizer:
                # print("join_str: {} not found in featurizer".format(join_str))
                return joins_vector
            joins_vector[self.join_featurizer[keys]] = 1.00
            return joins_vector
        else:
            keys = join_str.split("=")
            keys.sort()
            keys = ",".join(keys)
            joins_vector = np.zeros(len(self.join_featurizer))
            if keys not in self.join_featurizer:
                print("join_str: {} not found in featurizer".format(join_str))
                return joins_vector
            joins_vector[self.join_featurizer[keys]] = 1.00
            return joins_vector

    def get_pred_features(self, col, val, cmp_op,
            pred_est=None):

        if pred_est is not None:
            assert self.heuristic_features

        ## TODO: only difference is in computing pred_idx_start, otherwise both
        ## schemes seem same, so comine code + clean
        if self.featurization_type == "set":
            feat_idx_start = 0
            preds_vector = np.zeros(self.max_pred_len)
            if col not in self.featurizer:
                # print("col: {} not found in featurizer".format(col))
                return preds_vector

            assert col in self.column_stats
            # column one-hot value
            cidx = self.columns_onehot_idx[col]
            preds_vector[cidx] = 1.0
            feat_idx_start += len(self.columns_onehot_idx)

            cmp_op_idx, num_vals, continuous = self.featurizer[col]
            # set comparison operator 1-hot value, same for all types
            cmp_idx = self.cmp_ops_onehot[cmp_op]
            preds_vector[feat_idx_start + cmp_idx] = 1.00

            pred_idx_start = feat_idx_start + len(self.cmp_ops)
            col_info = self.column_stats[col]

            # 1 additional value for pg_est feature
            if pred_est:
                preds_vector[-1] = pred_est

            if not continuous:
                if self.separate_cont_bins:
                    pred_idx_start += self.continuous_feature_size

                if "like" in cmp_op:
                    assert len(val) == 1
                    num_buckets = min(self.max_discrete_featurizing_buckets,
                            col_info["num_values"])

                    # first half of spaces reserved for "IN" predicates
                    if self.separate_regex_bins:
                        pred_idx_start += num_buckets

                    regex_val = val[0].replace("%","")
                    pred_idx = deterministic_hash(regex_val) % num_buckets
                    preds_vector[pred_idx_start+pred_idx] = 1.00
                    for v in regex_val:
                        pred_idx = deterministic_hash(str(v)) % num_buckets
                        preds_vector[pred_idx_start+pred_idx] = 1.00

                    REGEX_USE_BIGRAMS = True
                    REGEX_USE_TRIGRAMS = True
                    if REGEX_USE_BIGRAMS:
                        for i,v in enumerate(regex_val):
                            if i != len(regex_val)-1:
                                pred_idx = deterministic_hash(v+regex_val[i+1]) % num_buckets
                                preds_vector[pred_idx_start+pred_idx] = 1.00

                    if REGEX_USE_TRIGRAMS:
                        for i,v in enumerate(regex_val):
                            if i < len(regex_val)-2:
                                pred_idx = deterministic_hash(v+regex_val[i+1]+ \
                                        regex_val[i+2]) % num_buckets
                                preds_vector[pred_idx_start+pred_idx] = 1.00

                    # FIXME: not sure if we should have this feature or not,
                    # just a number in a one-hot vector might be strange..
                    preds_vector[pred_idx_start + num_buckets] = len(regex_val)
                    if bool(re.search(r'\d', regex_val)):
                        preds_vector[pred_idx_start + num_buckets + 1] = 1

                else:
                    num_buckets = min(self.max_discrete_featurizing_buckets,
                            col_info["num_values"])
                    for v in val:
                        pred_idx = deterministic_hash(str(v)) % num_buckets
                        preds_vector[pred_idx_start+pred_idx] = 1.00
            else:
                # do min-max stuff
                # assert len(val) == 2
                min_val = float(col_info["min_value"])
                max_val = float(col_info["max_value"])
                min_max = [min_val, max_val]
                if isinstance(val, int):
                    cur_val = float(val)
                    norm_val = (cur_val - min_val) / (max_val - min_val)
                    norm_val = max(norm_val, 0.00)
                    norm_val = min(norm_val, 1.00)
                    preds_vector[pred_idx_start+0] = norm_val
                    preds_vector[pred_idx_start+1] = norm_val
                else:
                    for vi, v in enumerate(val):
                        if "literal" == v:
                            v = val["literal"]
                        # handling the case when one end of the range is
                        # missing
                        if v is None and vi == 0:
                            v = min_val
                        elif v is None and vi == 1:
                            v = max_val

                        if v == 'NULL' and vi == 0:
                            v = min_val
                        elif v == 'NULL' and vi == 1:
                            v = max_val

                        cur_val = float(v)
                        norm_val = (cur_val - min_val) / (max_val - min_val)
                        norm_val = max(norm_val, 0.00)
                        norm_val = min(norm_val, 1.00)
                        preds_vector[pred_idx_start+vi] = norm_val

            # if preds_vector[-1] != pred_est:
                # print(pred_est)
                # print("no match pred est")
                # pdb.set_trace()
            # assert preds_vector[-1] == pred_est

        else:
            preds_vector = np.zeros(self.pred_features_len)

            if col not in self.featurizer:
                print("col: {} not found in featurizer".format(col))
                return preds_vector

            # set comparison operator 1-hot value
            cmp_op_idx, num_vals, continuous = self.featurizer[col]
            cmp_idx = self.cmp_ops_onehot[cmp_op]
            preds_vector[cmp_op_idx+cmp_idx] = 1.00

            pred_idx_start = cmp_op_idx + len(self.cmp_ops)
            num_pred_vals = num_vals - len(self.cmp_ops)
            col_info = self.column_stats[col]
            # assert num_pred_vals >= 2

            # 1 additional value for pg_est feature
            # assert num_pred_vals <= col_info["num_values"] + 1

            ## FIXME: we are overshooting by one here
            if pred_est:
                # preds_vector[pred_idx_start + num_pred_vals] = pred_est
                preds_vector[pred_idx_start + num_pred_vals-1] = pred_est

            if not continuous:
                if "like" in cmp_op:
                    assert len(val) == 1
                    num_buckets = min(self.max_discrete_featurizing_buckets,
                            col_info["num_values"])

                    if self.separate_regex_bins:
                        # first half of spaces reserved for "IN" predicates
                        pred_idx_start += num_buckets

                    regex_val = val[0].replace("%","")
                    pred_idx = deterministic_hash(regex_val) % num_buckets
                    preds_vector[pred_idx_start+pred_idx] = 1.00
                    for v in regex_val:
                        pred_idx = deterministic_hash(str(v)) % num_buckets
                        preds_vector[pred_idx_start+pred_idx] = 1.00

                    REGEX_USE_BIGRAMS = True
                    REGEX_USE_TRIGRAMS = True
                    if REGEX_USE_BIGRAMS:
                        for i,v in enumerate(regex_val):
                            if i != len(regex_val)-1:
                                pred_idx = deterministic_hash(v+regex_val[i+1]) % num_buckets
                                preds_vector[pred_idx_start+pred_idx] = 1.00

                    if REGEX_USE_TRIGRAMS:
                        for i,v in enumerate(regex_val):
                            if i < len(regex_val)-2:
                                pred_idx = deterministic_hash(v+regex_val[i+1]+ \
                                        regex_val[i+2]) % num_buckets
                                preds_vector[pred_idx_start+pred_idx] = 1.00

                    preds_vector[pred_idx_start + num_buckets] = len(regex_val)
                    if bool(re.search(r'\d', regex_val)):
                        preds_vector[pred_idx_start + num_buckets + 1] = 1

                else:
                    num_buckets = min(self.max_discrete_featurizing_buckets,
                            col_info["num_values"])
                    for v in val:
                        pred_idx = deterministic_hash(str(v)) % num_buckets
                        preds_vector[pred_idx_start+pred_idx] = 1.00
            else:
                # do min-max stuff
                # assert len(val) == 2
                min_val = float(col_info["min_value"])
                max_val = float(col_info["max_value"])
                min_max = [min_val, max_val]
                if isinstance(val, int):
                    cur_val = float(val)
                    norm_val = (cur_val - min_val) / (max_val - min_val)
                    norm_val = max(norm_val, 0.00)
                    norm_val = min(norm_val, 1.00)
                    preds_vector[pred_idx_start+0] = norm_val
                    preds_vector[pred_idx_start+1] = norm_val
                else:
                    for vi, v in enumerate(val):
                        if "literal" == v:
                            v = val["literal"]
                        # handling the case when one end of the range is
                        # missing
                        if v is None and vi == 0:
                            v = min_val
                        elif v is None and vi == 1:
                            v = max_val

                        if v == 'NULL' and vi == 0:
                            v = min_val
                        elif v == 'NULL' and vi == 1:
                            v = max_val

                        cur_val = float(v)
                        norm_val = (cur_val - min_val) / (max_val - min_val)
                        norm_val = max(norm_val, 0.00)
                        norm_val = min(norm_val, 1.00)
                        preds_vector[pred_idx_start+vi] = norm_val

        return preds_vector

    def get_features(self, subgraph, true_sel=None):
        '''
        @subgraph:
        '''
        tables_vector = np.zeros(len(self.table_featurizer))
        preds_vector = np.zeros(self.pred_features_len)

        for nd in subgraph.nodes(data=True):
            node = nd[0]
            data = nd[1]
            tables_vector[self.table_featurizer[data["real_name"]]] = 1.00

            for i, col in enumerate(data["pred_cols"]):
                # add pred related feature
                val = data["pred_vals"][i]
                cmp_op = data["pred_types"][i]
                cmp_op_idx, num_vals, continuous = self.featurizer[col]
                cmp_idx = self.cmp_ops_onehot[cmp_op]
                preds_vector[cmp_op_idx+cmp_idx] = 1.00

                pred_idx_start = cmp_op_idx + len(self.cmp_ops)
                num_pred_vals = num_vals - len(self.cmp_ops)
                col_info = self.column_stats[col]
                # assert num_pred_vals >= 2
                assert num_pred_vals <= col_info["num_values"]
                if cmp_op == "in" or \
                        "like" in cmp_op or \
                        cmp_op == "eq":

                    if continuous:
                        assert len(val) <= self.continuous_feature_size
                        min_val = float(col_info["min_value"])
                        max_val = float(col_info["max_value"])
                        for vi, v in enumerate(val):
                            v = float(v)
                            normalized_val = (v - min_val) / (max_val - min_val)
                            preds_vector[pred_idx_start+vi] = 1.00
                    else:
                        num_buckets = min(self.max_discrete_featurizing_buckets,
                                col_info["num_values"])
                        assert num_pred_vals == num_buckets
                        # turn to 1 all the qualifying indexes in the 1-hot vector
                        if "like" in cmp_op:
                            print(cmp_op)
                            pdb.set_trace()
                        for v in val:
                            pred_idx = deterministic_hash(v) % num_buckets
                            preds_vector[pred_idx_start+pred_idx] = 1.00

                elif cmp_op in RANGE_PREDS:
                    assert cmp_op == "lt"

                    # does not have to be MIN / MAX, if there are very few values
                    # OR if the predicate is on string data, e.g., BETWEEN 'A' and 'F'

                    if not continuous:
                        # FIXME: temporarily, just treat it as discrete data
                        num_buckets = min(self.max_discrete_featurizing_buckets,
                                col_info["num_values"])
                        for v in val:
                            pred_idx = deterministic_hash(str(v)) % num_buckets
                            preds_vector[pred_idx_start+pred_idx] = 1.00
                    else:
                        # do min-max stuff
                        assert len(val) == 2

                        min_val = float(col_info["min_value"])
                        max_val = float(col_info["max_value"])
                        min_max = [min_val, max_val]
                        for vi, v in enumerate(val):
                            # handling the case when one end of the range is
                            # missing
                            if v is None and vi == 0:
                                v = min_val
                            elif v is None and vi == 1:
                                v = max_val

                            cur_val = float(v)
                            norm_val = (cur_val - min_val) / (max_val - min_val)
                            norm_val = max(norm_val, 0.00)
                            norm_val = min(norm_val, 1.00)
                            preds_vector[pred_idx_start+vi] = norm_val

        # based on edges, add the join conditions

        # TODO: combine all vectors, or not.
        if true_sel is not None:
            preds_vector[-1] = true_sel

        return preds_vector

    def unnormalize(self, y):
        if self.ynormalization == "log":
            est_card = np.exp((y + \
                self.min_val)*(self.max_val-self.min_val))
        elif self.ynormalization == "pg_total_selectivity":
            est_card = y*cards["total"]
        else:
            assert False
        return est_card

    def normalize_val(self, val, total):
        if self.ynormalization == "log":
            return (np.log(float(val)) - self.min_val) / (self.max_val-self.min_val)
        elif self.ynormalization == "selectivity":
            return float(val) / total
        else:
            assert False

    def _update_stats(self, qrep):
        '''
        '''
        if qrep["template_name"] not in self.templates:
            self.templates.append(qrep["template_name"])

        node_data = qrep["join_graph"].nodes(data=True)

        num_tables = len(node_data)
        if num_tables > self.max_tables:
            self.max_tables = num_tables

        num_preds = 0
        for node, info in node_data:
            num_preds += len(info["pred_cols"])

        if num_preds > self.max_preds:
            self.max_preds = num_preds

        num_joins = len(qrep["join_graph"].edges())
        if num_joins > self.max_joins:
            self.max_joins = num_joins

        cur_columns = []
        for node, info in qrep["join_graph"].nodes(data=True):
            for i, cmp_op in enumerate(info["pred_types"]):
                self.cmp_ops.add(cmp_op)
                if "like" in cmp_op:
                    self.regex_cols.add(info["pred_cols"][i])

            if node not in self.aliases:
                self.aliases[node] = info["real_name"]
                self.tables.add(info["real_name"])
            for col in info["pred_cols"]:
                cur_columns.append(col)

        joins = extract_join_clause(qrep["sql"])
        for join in joins:
            keys = join.split("=")
            keys.sort()
            keys = ",".join(keys)
            self.joins.add(keys)

        ## features required for plan-graph / flow-loss
        flow_start = time.time()
        subsetg = qrep["subset_graph"]
        node_list = list(subsetg.nodes())
        node_list.sort(key = lambda v: len(v))
        dest = node_list[-1]
        node_list.sort()
        info = {}
        tmp_name = qrep["template_name"]

        for node in subsetg.nodes():
            in_degree = subsetg.in_degree(node)
            if in_degree > self.max_in_degree:
                self.max_in_degree = in_degree

            out_degree = subsetg.out_degree(node)
            if out_degree > self.max_out_degree:
                self.max_out_degree = out_degree

            # TODO: compute flow / tolerances
            if tmp_name in self.template_info:
                continue
            info[node] = {}
            # paths from node -> dest, but edges are reversed in our
            # representation
            if self.feat_num_paths:
                all_paths = nx.all_simple_paths(subsetg, dest, node)
                num_paths = len(list(all_paths))
                if num_paths > self.max_paths:
                    self.max_paths = num_paths
                info[node]["num_paths"] = num_paths

        self.template_info[tmp_name] = info

        ## features required for each column used in the DB
        updated_cols = []
        for column in cur_columns:
            if column in self.column_stats:
                continue
            updated_cols.append(column)
            column_stats = {}
            table = column[0:column.find(".")]
            if table in self.aliases:
                table = ALIAS_FORMAT.format(TABLE = self.aliases[table],
                                    ALIAS = table)
            min_query = MIN_TEMPLATE.format(TABLE = table,
                                            COL   = column)
            max_query = MAX_TEMPLATE.format(TABLE = table,
                                            COL   = column)
            unique_count_query = UNIQUE_COUNT_TEMPLATE.format(FROM_CLAUSE = table,
                                                      COL = column)
            total_count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE = table)
            unique_vals_query = UNIQUE_VALS_TEMPLATE.format(FROM_CLAUSE = table,
                                                            COL = column)

            # TODO: move to using cached_execute
            column_stats[column] = {}
            column_stats[column]["min_value"] = self.execute(min_query)[0][0]
            column_stats[column]["max_value"] = self.execute(max_query)[0][0]
            column_stats[column]["num_values"] = \
                    self.execute(unique_count_query)[0][0]
            column_stats[column]["total_values"] = \
                    self.execute(total_count_query)[0][0]

            # only store all the values for tables with small alphabet
            # sizes (so we can use them for things like the PGM).
            # Otherwise, it bloats up the cache.
            if column_stats[column]["num_values"] <= 5000:
                column_stats[column]["unique_values"] = \
                        self.execute(unique_vals_query)
            else:
                column_stats[column]["unique_values"] = None

            self.column_stats.update(column_stats)

        if len(updated_cols) > 0:
            print("generated statistics for:" + ",".join(updated_cols))
