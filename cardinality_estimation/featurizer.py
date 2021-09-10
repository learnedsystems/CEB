import psycopg2 as pg
import pdb
import time
import random
from query_representation.utils import *

import time
from collections import OrderedDict, defaultdict
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

def qrep_to_df(qrep):
    '''
    Every subplan will be a row in the returned dataframe.
    Since each subplan can contain multiple tables / columns etc., we will have
    a unique column for each table/column accessed (FEAT_TYPE), and then it can
    have a bit-value to indicate presence, or a more complex value to indicate
    the predicate filter on that column etc.

    Format of dataframe column will be:
        {FEAT_TYPE}-{FEAT_NAME}
        table-table_name
        predicate_mi1-column_present
        predicate_ci-pg_est
        predicate_ci-filter_values

    Each such column name should map to a unique value
    '''
    pass

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

        # stats on the used columns
        #   table_name : column_name : attribute : value
        #   e.g., stats["title"]["id"]["max_value"] = 1010112
        #         stats["title"]["id"]["type"] = int
        #         stats["title"]["id"]["num_values"] = x
        self.column_stats = {}

        # will be set in setup()
        self.max_discrete_featurizing_buckets = None

        self.continuous_feature_size = 2

        # regex stuff
        self.ilike_bigrams = True
        self.ilike_trigrams = True

        self.featurizer = None
        self.cmp_ops = set()
        self.tables = set()
        self.joins = set()
        self.aliases = {}
        self.cmp_ops_onehot = {}
        self.regex_cols = set()

        self.templates = []

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

    def _init_pred_featurizer_combined(self):
        '''
        Reserves spots in the pred_feature vector for each column / kind of
        operator on it etc.
        '''
        self.featurizer = {}
        self.num_cols = len(self.column_stats)
        all_cols = list(self.column_stats.keys())
        all_cols.sort()

        self.columns_onehot_idx = {}
        for cidx, col_name in enumerate(all_cols):
            self.columns_onehot_idx[col_name] = cidx

        self.pred_features_len = 0
        # these comparator operator will be used for each predicate filter
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
            # for operator type of given column predicate
            pred_len += len(self.cmp_ops)

            # for pg_est of current table
            if self.heuristic_features:
                pred_len += 1

            # FIXME: special casing "id" not in col to avoid columns like
            # it.id; very specific to CEB workloads.
            if is_float(info["min_value"]) and is_float(info["max_value"]) \
                    and "id" not in col:
                # then use min-max normalization, no matter what
                # only support range-queries, so lower / and upper predicate
                pred_len += self.continuous_feature_size
                continuous = True
            else:
                # use 1-hot encoding
                continuous = False
                num_buckets = min(self.max_discrete_featurizing_buckets,
                        info["num_values"])
                pred_len += num_buckets

                if col in self.regex_cols:
                    # give it more space for #num-chars, #number in regex or
                    # not
                    pred_len += 2
                    if self.separate_ilike_bins:
                        # extra space for regex buckets
                        pred_len += num_buckets

            self.featurizer[col] = (self.pred_features_len, pred_len, continuous)
            self.pred_features_len += pred_len

            if self.max_pred_len < pred_len:
                self.max_pred_len = pred_len

        # for pg_est of the subplan as a whole
        if self.heuristic_features:
            self.pred_features_len += 1

        # for num_tables present
        if self.num_tables_feature:
            self.pred_features_len += 1

    def _init_pred_featurizer_set(self):
        assert self.featurization_type == "set"
        self.featurizer = {}
        self.num_cols = len(self.column_stats)
        all_cols = list(self.column_stats.keys())
        all_cols.sort()

        self.columns_onehot_idx = {}
        for cidx, col_name in enumerate(all_cols):
            self.columns_onehot_idx[col_name] = cidx

        self.pred_features_len = 0
        # these comparator operator will be used for each predicate filter
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

            # one-hot vector for which column is predicate on
            pred_len += self.num_cols

            # for num_tables present
            if self.num_tables_feature:
                pred_len += 1

            if self.heuristic_features:
                # for pg_est of current table in subplan
                pred_len += 1

                # for pg est of full subplan/query; repeated in each predicate
                # feature
                pred_len += 1

            # FIXME: special casing "id" not in col to avoid columns like
            # it.id; very specific to CEB workloads.
            if is_float(info["min_value"]) and is_float(info["max_value"]) \
                    and "id" not in col:
                # then use min-max normalization, no matter what
                # only support range-queries, so lower / and upper predicate
                pred_len += self.continuous_feature_size
                continuous = True
            else:
                # so the idxs used for continous features v/s categorical
                # features don't clash with each other
                if self.separate_cont_bins:
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
                    if self.separate_ilike_bins:
                        # extra space for regex buckets
                        pred_len += num_buckets

            self.featurizer[col] = (None, None, continuous)

            if self.max_pred_len < pred_len:
                self.max_pred_len = pred_len

    def setup(self,
            heuristic_features=True,
            ynormalization="log",
            table_features = True,
            pred_features = True,
            join_features = True,
            flow_features = False,
            num_tables_feature=False,
            separate_ilike_bins=True,
            separate_cont_bins=True,
            featurization_type="combined",
            max_discrete_featurizing_buckets=10,
            feat_num_paths= False, feat_flows=False,
            feat_pg_costs = True, feat_tolerance=False,
            feat_pg_path=True,
            feat_rel_pg_ests=True, feat_join_graph_neighbors=True,
            feat_rel_pg_ests_onehot=True,
            feat_pg_est_one_hot=True,
            cost_model=None, sample_bitmap=False, sample_bitmap_num=1000,
            sample_bitmap_buckets=1000,
            featkey=None):
        '''
        Sets up a transformation from a query subplan to three (table,
        predicates, join) 1d feature vectors based on the queries seen in
        update_column_stats. Optionally, will also setup similar feature
        mapping to a global `flow` 1d feature vector, which contains properties
        of the subplan-graph containing all subplanns in a query (see Flow-Loss paper for details).

        Important parameters:

            @featurization_type: `combined` or `set`. `combined` flattens a
            given subplan into a single 1d feature vector. In particular, this
            requires reserving a spot in this feature vector for each column
            used in the workloads; In contrast, `set` maps a given subplan into
            three sets (table, predicate, joins) of N feature vectors; If a
            subplan has three tables, five filter predicates, and two joins, it
            will map the subplan into a (3,TABLE_FEATURE_SIZE), (5,
            PREDICATE_FEATURE_SIZE), (2, JOIN_FEATURE_SIZE) vectors.
            Notice, for e.g., with predicates, the featurizer only needs to map
            a filter predicate on a single column to a feature vector; This is
            used in the set based MSCN architecture.  The main benefit is that
            it avoids potentially massively large flattened 1d feature vectors
            (which would require space M-columns x PREDICATE_FEATURE_SIZE). The
            drawback is that different subplans have different number of
            tables, predicates etc. And in order to get the MSCN implemenations
            working, we need to pad these with zeros so all `sets' are of same
            size, thus it requires A LOT of additional memory spent on just
            padding small subplans, e.g., a subplan which has 1 predicate
            filter will have size (1,PREDICATE_FEATURE_SIZE). But to pass it to
            the MSCN architecture, we'll need to pad it with zeros so its size
            will be (MAX_PREDICATE_FILTERS, PREDICATE_FEATURE_SIZE), where the
            MAX_PREDICATE_FILTERS is the largest value it is in the batch we
            pass into the MSCN model.

            @ynormalization: `log` or `selectivity`.

            Other features areflags for various minor tweaks / additional
            features. For most use cases, the default values should suffice.

        This Generates a featurizer dict:
            {table_name: (idx, num_vals)}
            {join_key: (idx, num_vals)}
            {pred_column: (idx, num_vals)}
        where the idx refers to the elements position in the feature vector,
        and num_vals refers to the number of values it will occupy; The
        implementations of `combined` or `set` featurization type uses these
        dictionaries slightly differently for predicates; see the
        implementations in get_pred_features for more details.
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

        if self.featurization_type == "combined":
            self._init_pred_featurizer_combined()
        elif self.featurization_type == "set":
            self._init_pred_featurizer_set()

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
            self.num_flow_features += 1

    def _handle_continuous_feature(self, pfeats, pred_idx_start,
            col, val):
        '''
        '''
        col_info = self.column_stats[col]
        min_val = float(col_info["min_value"])
        max_val = float(col_info["max_value"])

        assert isinstance(val, list)
        for vi, v in enumerate(val):
            # assert v is not None
            # handling the case when one end of the range predicate is
            # missing
            if (v is None or v == "NULL"):
                if vi == 0:
                    v = min_val
                else:
                    v = max_val

            # use min-max normalization for continuous features
            cur_val = float(v)
            norm_val = (cur_val - min_val) / (max_val - min_val)
            norm_val = max(norm_val, 0.00)
            norm_val = min(norm_val, 1.00)
            pfeats[pred_idx_start+vi] = norm_val

    def _handle_categorical_feature(self, pfeats, pred_idx_start,
            col, val):
        '''
        hashed features;
        '''
        col_info = self.column_stats[col]
        num_buckets = min(self.max_discrete_featurizing_buckets,
                col_info["num_values"])
        for v in val:
            pred_idx = deterministic_hash(str(v)) % num_buckets
            pfeats[pred_idx_start+pred_idx] = 1.00

    def _get_pg_est(self, subpinfo):
        # subpinfo = subsetgraph.nodes()[node]
        pg_est = subpinfo[self.ckey]["expected"]
        # note: total is only needed for self.ynormalization == selectivity
        if "total" in subpinfo[self.ckey]:
            total = subpinfo[self.ckey]["total"]
        else:
            total = None
        subp_est = self.normalize_val(pg_est,
                total)
        return subp_est

    def _handle_ilike_feature(self, pfeats, pred_idx_start,
            col, val):
        assert len(val) == 1
        col_info = self.column_stats[col]
        num_buckets = min(self.max_discrete_featurizing_buckets,
                col_info["num_values"])

        if self.separate_ilike_bins:
            # first half of spaces reserved for "IN" predicates
            pred_idx_start += num_buckets

        regex_val = val[0].replace("%","")
        pred_idx = deterministic_hash(regex_val) % num_buckets
        pfeats[pred_idx_start+pred_idx] = 1.00
        for v in regex_val:
            pred_idx = deterministic_hash(str(v)) % num_buckets
            pfeats[pred_idx_start+pred_idx] = 1.00

        if self.ilike_bigrams:
            for i,v in enumerate(regex_val):
                if i != len(regex_val)-1:
                    pred_idx = deterministic_hash(v+regex_val[i+1]) % num_buckets
                    pfeats[pred_idx_start+pred_idx] = 1.00

        if self.ilike_trigrams:
            for i,v in enumerate(regex_val):
                if i < len(regex_val)-2:
                    pred_idx = deterministic_hash(v+regex_val[i+1]+ \
                            regex_val[i+2]) % num_buckets
                    pfeats[pred_idx_start+pred_idx] = 1.00

        pfeats[pred_idx_start + num_buckets] = len(regex_val)

        # regex has num or not feature
        if bool(re.search(r'\d', regex_val)):
            pfeats[pred_idx_start + num_buckets + 1] = 1

    def get_subplan_features_set(self, qrep, subplan):
        '''
        @ret: {}
            key: table,pred,join etc.
            val: [[feats1], [feats2], ...]
            Note that there can be a variable number of arrays depending on the
            subplan we're trying to featurize.
        '''
        assert isinstance(subplan, tuple)
        featdict = {}
        subsetgraph = qrep["subset_graph"]
        joingraph = qrep["join_graph"]

        alltablefeats = []
        if self.table_features:
            ## table features
            # loop over each node, update the tfeats bitvector
            for alias in subplan:
                tfeats = np.zeros(self.table_features_len)
                # need to find its real table name from the join_graph
                table = joingraph.nodes()[alias]["real_name"]
                if table not in self.table_featurizer:
                    print("table: {} not found in featurizer".format(table))
                    continue
                # Note: same table might be set to 1.0 twice, in case of aliases
                tfeats[self.table_featurizer[table]] = 1.00
                alltablefeats.append(tfeats)

        featdict["table"] = alltablefeats

        alljoinfeats = []
        if self.join_features:
            ## join features
            seenjoins = set()
            for alias1 in subplan:
                for alias2 in subplan:
                    ekey = (alias1, alias2)
                    if ekey in joingraph.edges():
                        join_str = joingraph.edges()[ekey]["join_condition"]
                        if join_str in seenjoins:
                            continue
                        seenjoins.add(join_str)
                        jfeats  = np.zeros(len(self.joins))
                        keys = join_str.split("=")
                        keys.sort()
                        keys = ",".join(keys)
                        if keys not in self.join_featurizer:
                            print("join_str: {} not found in featurizer".format(join_str))
                            continue
                        jfeats[self.join_featurizer[keys]] = 1.00
                        alljoinfeats.append(jfeats)

            if len(alljoinfeats) == 0:
                alljoinfeats.append(np.zeros(len(self.joins)))

        featdict["join"] = alljoinfeats

        allpredfeats = []
        if self.pred_features:
            for alias in subplan:
                aliasinfo = joingraph.nodes()[alias]
                if len(aliasinfo["pred_cols"]) == 0:
                    continue
                col = aliasinfo["pred_cols"][0]
                val = aliasinfo["pred_vals"][0]
                # FIXME: should handle this at the level of parsing
                if isinstance(val, dict):
                    val = val["literal"]
                cmp_op = aliasinfo["pred_types"][0]

                feat_idx_start = 0
                pfeats = np.zeros(self.max_pred_len)
                if col not in self.featurizer:
                    print("col: {} not found in featurizer".format(col))
                    continue

                assert col in self.column_stats
                # which column does the current feature belong to
                cidx = self.columns_onehot_idx[col]
                pfeats[cidx] = 1.0

                feat_idx_start += len(self.columns_onehot_idx)

                _, _, continuous = self.featurizer[col]

                # set comparison operator 1-hot value, same for all types
                cmp_idx = self.cmp_ops_onehot[cmp_op]
                pfeats[feat_idx_start + cmp_idx] = 1.00

                pred_idx_start = feat_idx_start + len(self.cmp_ops)
                col_info = self.column_stats[col]

                if continuous:
                    self._handle_continuous_feature(pfeats, pred_idx_start,
                            col, val)
                else:
                    if self.separate_cont_bins:
                        pred_idx_start += self.continuous_feature_size

                    if "like" in cmp_op:
                        self._handle_ilike_feature(pfeats, pred_idx_start,
                                col, val)
                    else:
                        self._handle_categorical_feature(pfeats, pred_idx_start,
                                col, val)

                # add the appropriate postgresql estimate for this table in the
                # subplan; Note that the last elements are reserved for the
                # heuristic estimates for both continuous / categorical
                # features
                if self.heuristic_features:
                    node_key = tuple([alias])
                    alias_est = self._get_pg_est(subsetgraph.nodes()[node_key])
                    assert pfeats[-1] == 0.0
                    assert pfeats[-2] == 0.0
                    pfeats[-2] = alias_est
                    subp_est = self._get_pg_est(subsetgraph.nodes()[subplan])
                    pfeats[-1] = subp_est

                allpredfeats.append(pfeats)

            if len(allpredfeats) == 0:
                allpredfeats.append(np.zeros(self.max_pred_len))

        featdict["pred"] = allpredfeats
        flow_features = []

        if self.flow_features:
            flow_features = self.get_flow_features(subplan,
                    qrep["subset_graph"], qrep["template_name"],
                    qrep["join_graph"])
        featdict["flow"] = flow_features

        return featdict

    def get_subplan_features_combined(self, qrep, subplan):
        assert isinstance(subplan, tuple)
        featvectors = []

        # we need the joingraph here because all the information about
        # predicate filters etc. on each of the individual tables is stored in
        # the joingraph; subsetgraph stores just the names of the
        # tables/aliases involved in a join
        subsetgraph = qrep["subset_graph"]
        joingraph = qrep["join_graph"]

        if self.table_features:
            tfeats = np.zeros(self.table_features_len)
            ## table features
            # loop over each node, update the tfeats bitvector
            for alias in subplan:
                # need to find its real table name from the join_graph
                table = joingraph.nodes()[alias]["real_name"]
                if table not in self.table_featurizer:
                    print("table: {} not found in featurizer".format(table))
                    continue
                # Note: same table might be set to 1.0 twice, in case of aliases
                tfeats[self.table_featurizer[table]] = 1.00
            featvectors.append(tfeats)

        if self.join_features:
            ## join features
            jfeats  = np.zeros(len(self.joins))
            for alias1 in subplan:
                for alias2 in subplan:
                    ekey = (alias1, alias2)
                    if ekey in joingraph.edges():
                        join_str = joingraph.edges()[ekey]["join_condition"]
                        keys = join_str.split("=")
                        keys.sort()
                        keys = ",".join(keys)
                        if keys not in self.join_featurizer:
                            print("join_str: {} not found in featurizer".format(join_str))
                            continue
                        jfeats[self.join_featurizer[keys]] = 1.00
            featvectors.append(jfeats)

        ## predicate filter features
        if self.pred_features:
            pfeats = np.zeros(self.pred_features_len)
            for alias in subplan:
                aliasinfo = joingraph.nodes()[alias]
                if len(aliasinfo["pred_cols"]) == 0:
                    continue
                # FIXME: only supporting 1 predicate per column right now ---
                # that's all we had in CEB. Supporting an arbitrary number of
                # predicates can be messy with a fixed featurization scheme to
                # flatten into a 1d array; Presumably, this assumes a known
                # workload, and so we could `reserve` additional spaces for each
                # known predicate on a column
                col = aliasinfo["pred_cols"][0]
                val = aliasinfo["pred_vals"][0]
                # FIXME: should handle this at the level of parsing
                if isinstance(val, dict):
                    val = val["literal"]
                cmp_op = aliasinfo["pred_types"][0]

                if col not in self.featurizer:
                    print("col: {} not found in featurizer".format(col))
                    continue

                cmp_op_idx, num_vals, continuous = self.featurizer[col]
                cmp_idx = self.cmp_ops_onehot[cmp_op]
                pfeats[cmp_op_idx+cmp_idx] = 1.00
                pred_idx_start = cmp_op_idx + len(self.cmp_ops)

                if continuous:
                    self._handle_continuous_feature(pfeats, pred_idx_start,
                            col, val)
                else:
                    if "like" in cmp_op:
                        self._handle_ilike_feature(pfeats, pred_idx_start,
                                col, val)
                    else:
                        self._handle_categorical_feature(pfeats, pred_idx_start,
                                col, val)

                # remaining values after the cmp_op feature
                num_pred_vals = num_vals - len(self.cmp_ops)
                # add the appropriate postgresql estimate for this table in the
                # subplan
                if self.heuristic_features:
                    assert pfeats[pred_idx_start + num_pred_vals-1] == 0.0
                    node_key = tuple([alias])
                    subp_est = self._get_pg_est(subsetgraph.nodes()[node_key])
                    pfeats[pred_idx_start + num_pred_vals-1] = subp_est

            # Add the postgres heuristic estimate for the whole subplan as a
            # feature to the predicate feature vector.
            if self.heuristic_features:
                subp_est = self._get_pg_est(subsetgraph.nodes()[subplan])
                assert pfeats[-1] == 0.0
                pfeats[-1] = subp_est

            featvectors.append(pfeats)

        if self.flow_features:
            flow_features = self.get_flow_features(subplan,
                    qrep["subset_graph"], qrep["template_name"],
                    qrep["join_graph"])
            featvectors.append(flow_features)

        feat = np.concatenate(featvectors)
        return feat

    def get_subplan_features(self, qrep, node):
        '''
        @subsetg:
        @node: subplan in the subsetgraph;
        @ret: []
            will depend on if self.featurization_type == set or combined;
        '''
        if self.sample_bitmap:
            assert False, "TODO: not implemented yet"

        # the shapes will depend on combined v/s set feat types
        if self.featurization_type == "combined":
            x = self.get_subplan_features_combined(qrep,
                    node)
        elif self.featurization_type == "set":
            x = self.get_subplan_features_set(qrep,
                    node)
        else:
            assert False

        cardinfo = qrep["subset_graph"].nodes()[node]
        # Y, and cur_info
        true_val = cardinfo[self.ckey]["actual"]
        if "total" in cardinfo[self.ckey]:
            total = cardinfo[self.ckey]["total"]
        else:
            total = None
        y = self.normalize_val(true_val, total)

        return x,y

    def get_onehot_bucket(self, num_buckets, base, val):
        assert val >= 1.0
        for i in range(num_buckets):
            if val > base**i and val < base**(i+1):
                return i

        return num_buckets

    def get_flow_features(self, node, subsetg,
            template_name, join_graph):
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

        if self.feat_pg_costs and self.heuristic_features and \
                self.cost_model is not None:
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

        if self.feat_rel_pg_ests and self.heuristic_features \
                and self.cost_model is not None:
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
                and self.heuristic_features \
                and self.cost_model is not None:
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

        return flow_features

    def unnormalize(self, y):
        if self.ynormalization == "log":
            est_card = np.exp((y + \
                self.min_val)*(self.max_val-self.min_val))
        elif self.ynormalization == "selectivity":
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
