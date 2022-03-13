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
import pickle
import copy

JOIN_KEY_MAX_TMP = """SELECT COUNT(*), {COL} FROM {TABLE} GROUP BY {COL} ORDER BY COUNT(*) DESC LIMIT 1"""

JOIN_KEY_MIN_TMP = """SELECT COUNT(*), {COL} FROM {TABLE} GROUP BY {COL} ORDER BY COUNT(*) ASC LIMIT 1"""

JOIN_KEY_AVG_TMP = """SELECT AVG(count) FROM (SELECT COUNT(*) AS count, {COL} FROM {TABLE} GROUP BY {COL} ORDER BY COUNT(*)) AS tmp"""

JOIN_KEY_VAR_TMP = """SELECT VARIANCE(count) FROM (SELECT COUNT(*) AS count, {COL} FROM {TABLE} GROUP BY {COL} ORDER BY COUNT(*)) AS tmp"""

JOIN_KEY_COUNT_TMP = """SELECT COUNT({COL}) FROM {TABLE}"""
JOIN_KEY_DISTINCT_TMP = """SELECT COUNT(DISTINCT {COL}) FROM {TABLE}"""

# TODO:
NULL_FRAC_TMP = """SELECT null_frac FROM pg_stats WHERE tablename='{TABLE}' AND attname = '{COL}'"""

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

MCV_TEMPLATE= """SELECT most_common_vals,most_common_freqs FROM pg_stats WHERE tablename = '{TABLE}' and attname = '{COL}'"""

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

import string

def preprocess_word(word, exclude_nums=False, exclude_the=False,
        exclude_words=[], min_len=0):
    word = str(word)
    # no punctuation
    exclude = set(string.punctuation)
    # exclude the as well
    if exclude_the:
        exclude.add("the")
    if exclude_nums:
        for i in range(10):
            exclude.add(str(i))

    # exclude.remove("%")
    word = ''.join(ch for ch in word if ch not in exclude)

    # make it lowercase
    word = word.lower()
    final_words = []

    for w in word.split():
        if w in exclude_words:
            continue
        if len(w) < min_len:
            continue
        final_words.append(w)

    return " ".join(final_words)

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

        self.regex_templates = set()
        # stats on the used columns
        #   table_name : column_name : attribute : value
        #   e.g., stats["title"]["id"]["max_value"] = 1010112
        #         stats["title"]["id"]["type"] = int
        #         stats["title"]["id"]["num_values"] = x
        self.column_stats = {}

        self.mcvs = {}

        self.join_key_stats = {}
        self.primary_join_keys = set()
        self.join_key_normalizers = {}
        self.join_key_stat_names = ["null_frac", "count", "distinct",
                "avg_key", "var_key", "max_key", "min_key"]
        self.join_key_stat_tmps = [NULL_FRAC_TMP, JOIN_KEY_COUNT_TMP,
                JOIN_KEY_DISTINCT_TMP, JOIN_KEY_AVG_TMP, JOIN_KEY_VAR_TMP,
                JOIN_KEY_MAX_TMP, JOIN_KEY_MIN_TMP]

        # debug version
        # self.join_key_stat_names = ["max_key"]
        # self.join_key_stat_tmps = [JOIN_KEY_MAX_TMP]

        # self.column_key_stats = {}
        # self.column_stat_names = ["count", "distinct", "avg_key", "var_key", "max_key", "min_key"]
        # self.column_stat_tmps = [JOIN_KEY_COUNT_TMP, JOIN_KEY_DISTINCT_TMP, JOIN_KEY_AVG_TMP, JOIN_KEY_VAR_TMP, JOIN_KEY_MAX_TMP, JOIN_KEY_MIN_TMP]
        # self.column_key_normalizers = {}

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
        self.max_pred_vals = 0

    def update_column_stats(self, qreps):
        for qrep in qreps:
            self._update_stats(qrep)

        for si,statname in enumerate(self.join_key_stat_names):
            allvals = []

            for col,kvs in self.join_key_stats.items():
                if statname not in kvs:
                    continue
                allvals.append(kvs[statname])
            if len(allvals) == 0:
                continue
            self.join_key_normalizers[statname] = (np.mean(allvals),
                    np.std(allvals))

    def update_seen_preds(self, qreps):
        # key: column name, val: set() of seen values
        self.seen_preds = {}
        # need separate dictionaries, because like constants / like
        # featurization has very different meaning from categorical
        # featurization
        self.seen_like_preds = {}
        for qrep in qreps:
            for node, info in qrep["join_graph"].nodes(data=True):
                for ci, col in enumerate(info["pred_cols"]):
                    # cur_columns.append(col)
                    if not self.feat_separate_alias:
                        col = ''.join([ck for ck in col if not ck.isdigit()])

                    if col not in self.seen_preds:
                        self.seen_preds[col] = set()

                    vals = info["pred_vals"][ci]
                    cmp_op = info["pred_types"][ci]
                    for val in vals:
                        self.seen_preds[col].add(val)

                    if "like" in cmp_op:
                        if col not in self.seen_like_preds:
                            self.seen_like_preds[col] = set()
                        assert len(vals) == 1
                        self.seen_like_preds[col].add(vals[0])

    def update_using_saved_stats(self, featdata):
        for k,v in featdata.items():
            setattr(self, k, v)

        # reset these using workload
        self.max_tables = 0
        self.max_joins = 0
        self.max_preds = 0
        self.max_pred_vals = 0

        self.cmp_ops = set()
        self.joins = set()
        self.tables = set()
        self.regex_cols = set()
        self.aliases = {}
        self.regex_templates = set()

    def update_workload_stats(self, qreps):
        for qrep in qreps:
            if qrep["template_name"] not in self.templates:
                self.templates.append(qrep["template_name"])

            node_data = qrep["join_graph"].nodes(data=True)

            num_tables = len(node_data)
            if num_tables > self.max_tables:
                self.max_tables = num_tables

            num_preds = 0
            num_pred_vals = 0
            for node, info in node_data:
                num_preds += len(info["pred_cols"])
                if len(info["pred_vals"]) == 0:
                    continue

                if isinstance(info["pred_vals"][0], list):
                    num_pred_vals += len(info["pred_vals"][0])
                else:
                    num_pred_vals += len(info["pred_vals"])

            if num_preds > self.max_preds:
                self.max_preds = num_preds

            if num_pred_vals > self.max_pred_vals:
                self.max_pred_vals = num_pred_vals
                # print(num_pred_vals)
                # pdb.set_trace()

            num_joins = len(qrep["join_graph"].edges())
            if num_joins > self.max_joins:
                self.max_joins = num_joins

            cur_columns = []
            for node, info in qrep["join_graph"].nodes(data=True):
                for i, cmp_op in enumerate(info["pred_types"]):
                    self.cmp_ops.add(cmp_op)
                    if "like" in cmp_op:
                        self.regex_cols.add(info["pred_cols"][i])
                        self.regex_templates.add(qrep["template_name"])

                if node not in self.aliases:
                    self.aliases[node] = info["real_name"]
                    self.tables.add(info["real_name"])
                for col in info["pred_cols"]:
                    cur_columns.append(col)

            joins = extract_join_clause(qrep["sql"])
            for joinstr in joins:
                # get rid of whitespace
                joinstr = joinstr.replace(" ", "")
                if not self.feat_separate_alias:
                    joinstr = ''.join([ck for ck in joinstr if not ck.isdigit()])

                keys = joinstr.split("=")
                keys.sort()
                keys = ",".join(keys)
                self.joins.add(keys)

        print("max pred vals: {}".format(self.max_pred_vals))

    def update_ystats(self, qreps):
        y = np.array(get_all_cardinalities(qreps, self.ckey))
        if self.ynormalization == "log":
            y = np.log(y)

        self.max_val = np.max(y)
        self.min_val = np.min(y)

    def execute(self, sql):
        '''
        '''
        try:
            con = pg.connect(user=self.user, host=self.db_host, port=self.port,             password=self.pwd, dbname=self.db_name)
        except:
            con = pg.connect(user=self.user, port=self.port, password=self.pwd, dbname=self.db_name)


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

            if self.feat_mcvs:
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
                    pred_len += self.max_like_featurizing_buckets

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
        # self.featurizer = {}
        # self.featurizer_type_idxs = {}

        self.num_cols = len(self.column_stats)
        all_cols = list(self.column_stats.keys())
        all_cols.sort()

        self.columns_onehot_idx = {}
        for cidx, col_name in enumerate(all_cols):
            self.columns_onehot_idx[col_name] = cidx

        # these comparator operator will be used for each predicate filter
        for i, cmp_op in enumerate(sorted(self.cmp_ops)):
            self.cmp_ops_onehot[cmp_op] = i

        ## TODO: separate these out in a different function.
        # to find the number of features, need to go over every column, and
        # choose how many spots to keep for them
        col_keys = list(self.column_stats.keys())
        col_keys.sort()

        pred_len = 0
        self.featurizer_type_idxs["cmp_op"] = (pred_len, len(self.cmp_ops))
        pred_len += len(self.cmp_ops)

        use_onehot = "onehot" in self.set_column_feature
        use_stats = "stats" in self.set_column_feature

        if use_onehot:
            self.set_column_features_len = len(self.column_stats)
            self.featurizer_type_idxs["col_onehot"] = (pred_len,
                    len(self.column_stats))
            pred_len += len(self.column_stats)

        if use_stats:
            self.set_column_features_len = 0
            # number of filters in the column
            # self.set_column_features_len += 1
            # TODO: think of other ways
            self.set_column_features_len += len(self.join_key_stat_names)
            self.featurizer_type_idxs["col_stats"] = (pred_len,
                    len(self.join_key_stat_names))
            pred_len += len(self.join_key_stat_names)

        else:
            self.set_column_features_len = 0

        if self.num_tables_feature:
            self.featurizer_type_idxs["num_tables"] = (pred_len, 1)
            pred_len += 1

        self.featurizer_type_idxs["constant_continuous"] = (pred_len,
                self.continuous_feature_size)
        pred_len += self.continuous_feature_size

        ilike_feat_size = 2 + self.max_like_featurizing_buckets
        self.featurizer_type_idxs["constant_like"] = (pred_len,
                ilike_feat_size)
        pred_len += ilike_feat_size

        # for discrete features
        self.featurizer_type_idxs["constant_discrete"] = (pred_len,
                self.max_discrete_featurizing_buckets)
        pred_len += self.max_discrete_featurizing_buckets

        if self.embedding_fn is not None:
            self.featurizer_type_idxs["constant_embedding"] = (pred_len,
                    self.embedding_size)
            pred_len += self.embedding_size

        if self.heuristic_features:
            self.featurizer_type_idxs["heuristic_ests"] = (pred_len, 2)
            # for pg_est of current table in subplan
            # and for pg est of full subplan/query; repeated in each predicate
            # feature
            pred_len += 2

        self.max_pred_len = pred_len
        self.pred_onehot_mask = np.ones(self.max_pred_len)
        a,b = self.featurizer_type_idxs["constant_discrete"]
        self.pred_onehot_mask[a:a+b] = 0.0

        a,b = self.featurizer_type_idxs["constant_like"]
        self.pred_onehot_mask[a:a+b] = 0.0

        if "col_onehot" in self.featurizer_type_idxs:
            a,b = self.featurizer_type_idxs["col_onehot"]
            self.pred_onehot_mask[a:a+b] = 0.0

        ## mapping columns to continuous or not
        for col in col_keys:
            info = self.column_stats[col]
            if is_float(info["min_value"]) and is_float(info["max_value"]) \
                    and "id" not in col:
                # then use min-max normalization, no matter what
                # only support range-queries, so lower / and upper predicate
                # continuous = True
                self.column_stats[col]["continuous"] = True
            else:
                self.column_stats[col]["continuous"] = False

        # for col in col_keys:
            # info = self.column_stats[col]
            # pred_len = 0
            # # for operator type
            # pred_len += len(self.cmp_ops)

            # if self.set_column_feature == "onehot":
                # # one-hot vector for which column is predicate on
                # pred_len += self.num_cols
            # elif self.set_column_feature == "stats":
                # pred_len += self.set_column_features_len
            # else:
                # pass

            # # for num_tables present
            # if self.num_tables_feature:
                # pred_len += 1

            # # these are stored at the very end;
            # if self.heuristic_features:
                # # for pg_est of current table in subplan
                # pred_len += 1
                # # for pg est of full subplan/query; repeated in each predicate
                # # feature
                # pred_len += 1

            # # FIXME: special casing "id" not in col to avoid columns like
            # # it.id; very specific to CEB workloads.
            # if is_float(info["min_value"]) and is_float(info["max_value"]) \
                    # and "id" not in col:
                # # then use min-max normalization, no matter what
                # # only support range-queries, so lower / and upper predicate
                # pred_len += self.continuous_feature_size
                # continuous = True
            # else:
                # # so the idxs used for continous features v/s categorical
                # # features don't clash with each other
                # pred_len += self.continuous_feature_size
                # pred_len += 2
                # pred_len += self.max_like_featurizing_buckets

                # # for discrete features
                # pred_len += self.max_discrete_featurizing_buckets
                # continuous = False

                # if self.embedding_fn is not None:
                    # pred_len += self.embedding_size

            # if self.max_pred_len < pred_len:
                # self.max_pred_len = pred_len
                # print("self.max_pred_len updated to: ", pred_len)

    def setup(self,
            heuristic_features=True,
            feat_separate_alias=True,
            separate_ilike_bins=False,
            separate_cont_bins=False,
            onehot_dropout=False,
            ynormalization="log",
            table_features = True,
            pred_features = True,
            feat_onlyseen_preds = True,
            seen_preds = False,
            set_column_feature = "onehot",
            join_features = "onehot",
            flow_features = False,
            embedding_fn = None,
            embedding_pooling = None,
            num_tables_feature=False,
            featurization_type="combined",
            max_discrete_featurizing_buckets=10,
            max_like_featurizing_buckets=10,
            feat_num_paths= False, feat_flows=False,
            feat_pg_costs = True, feat_tolerance=False,
            feat_pg_path=True,
            feat_rel_pg_ests=True, feat_join_graph_neighbors=True,
            feat_rel_pg_ests_onehot=True,
            feat_pg_est_one_hot=True,
            feat_mcvs = False,
            implied_pred_features=False,
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

        if self.embedding_fn == "none":
            self.embedding_fn = None

        if self.set_column_feature == "1":
            self.set_column_feature = "onehot"

        # change the
        if not self.feat_separate_alias:
            newstats = {}
            for k,v in self.column_stats.items():
                newk = ''.join([ck for ck in k if not ck.isdigit()])
                newstats[newk] = v
            self.column_stats = newstats

            jkey_stats = {}
            for k,v in self.join_key_stats.items():
                newk = ''.join([ck for ck in k if not ck.isdigit()])
                jkey_stats[newk] = v
            self.join_key_stats = jkey_stats

            print("Updated stats to remove alias based columns")


    def init_feature_mapping(self):

        if self.embedding_fn is not None:
            assert os.path.exists(self.embedding_fn), \
                    "Embeddings file not found: %s" % self.embedding_fn
            print(self.embedding_fn)
            with open(self.embedding_fn, 'rb') as handle:
                self.embeddings = pickle.load(handle)
            print("Loaded embeddings, with %d entries" % len(self.embeddings))

            sample_embedding = list(self.embeddings.values())[0]
            self.embedding_size = len(sample_embedding)
            print("forcing discrete buckets = 1, because we are using embeddings")
            self.max_discrete_featurizing_buckets = 1

            # FIXME: will depend on what embedding pooling scheme we choose
            assert self.max_pred_vals > 0
            assert self.max_pred_vals >= self.max_preds

            if self.embedding_pooling == "sum":
                pass
            else:
                self.max_preds = self.max_pred_vals

        self.featurizer_type_idxs = {}

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

        # only have onehot encoding for tables
        self.table_onehot_mask = np.zeros(self.table_features_len)

        ## join features
        if self.join_features == "1":
            # or table one
            self.join_features = "onehot"
        elif self.join_features == "0":
            self.join_features = False

        use_onehot = "onehot" in self.join_features
        use_stats = "stats" in self.join_features

        self.join_features_len = 0
        if use_onehot:
            self.join_featurizer = {}
            for i, join in enumerate(sorted(self.joins)):
                self.join_featurizer[join] = i
            self.join_features_len += len(self.joins)
            self.featurizer_type_idxs["join_onehot"] = (0, len(self.joins))

        if use_stats:
            joinstats_len = 0
            # self.join_features_len = 0
            # primary key : foreign key OR fk : fk (onehot)
            joinstats_len += 2
            # is self-join or not (boolean)
            joinstats_len  += 1
            # dv(tab1) / dv(tab2)
            joinstats_len += 1
            # for both side of joins; pk first OR just sort;
            joinstats_len += len(self.join_key_stat_names)*2

            self.featurizer_type_idxs["join_stats"] = (self.join_features_len,
                                                    joinstats_len)

            self.join_features_len += joinstats_len

        self.join_onehot_mask = np.ones(self.join_features_len)
        if "join_onehot" in self.featurizer_type_idxs:
            a,b = self.featurizer_type_idxs["join_onehot"]
            self.join_onehot_mask[a:b] = 0.0

        # if self.join_features == "onehot":
            # self.join_featurizer = {}
            # for i, join in enumerate(sorted(self.joins)):
                # self.join_featurizer[join] = i

        # if self.join_features == "onehot" \
                # or self.join_features == "1":
            # self.join_features_len = len(self.joins)
        # elif self.join_features == "onehot_tables":
            # self.join_features_len = len(self.tables)
        # elif self.join_features == "onehot_debug":
            # self.join_features_len = len(self.tables)

        # elif self.join_features == "stats":
            # self.join_features_len = 0
            # # primary key : foreign key OR fk : fk (onehot)
            # self.join_features_len += 2
            # # is self-join or not (boolean)
            # self.join_features_len += 1
            # # dv(tab1) / dv(tab2)
            # self.join_features_len += 1

            # # for both side of joins; pk first OR just sort;
            # self.join_features_len += len(self.join_key_stat_names)*2


        ## predicate filter features
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

    def _handle_categorical_feature(self, pfeats,
            pred_idx_start, col, val):
        '''
        hashed features;
        '''
        col_info = self.column_stats[col]
        if self.featurization_type == "combined":
            num_buckets = min(self.max_like_featurizing_buckets,
                    col_info["num_values"])
        else:
            num_buckets = self.max_discrete_featurizing_buckets

        if self.feat_mcvs and col in self.mcvs:
            mcvs = self.mcvs[col]
            total_count = 0
            for predicate in val:
                if predicate in mcvs:
                    total_count += mcvs[predicate]

            assert pfeats[-3] == 0.0
            if total_count != 0:
                norm_count = self.normalize_val(total_count, None)
                pfeats[-3] = norm_count

        if self.embedding_fn is not None:
            for predicate in val:
                words = preprocess_word(predicate)
                valkey = str(col) + str(words)
                if valkey in self.embeddings:
                    embedding = self.embeddings[valkey]
                    pred_end = pred_idx_start+self.embedding_size
                    pfeats[pred_idx_start:pred_end] = embedding

        else:
            if self.feat_onlyseen_preds:
                if col not in self.seen_preds:
                    # unseen column
                    return

            for v in val:
                if self.feat_onlyseen_preds:
                    if v not in self.seen_preds[col]:
                        continue

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

        if self.feat_onlyseen_preds:
            if col not in self.seen_like_preds:
                return

        col_info = self.column_stats[col]

        if self.featurization_type == "combined":
            num_buckets = min(self.max_like_featurizing_buckets,
                    col_info["num_values"])
        else:
            num_buckets = self.max_like_featurizing_buckets

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

    def _handle_join_features(self, join_str):
        join_str = join_str.replace(" ", "")

        use_onehot = "onehot" in self.join_features
        use_stats = "stats" in self.join_features
        jfeats  = np.zeros(self.join_features_len)

        if use_onehot:
            if not self.feat_separate_alias:
                join_str2 = ''.join([ck for ck in join_str if not ck.isdigit()])
            else:
                join_str2 = join_str

            keys = join_str2.split("=")
            keys.sort()
            keys = ",".join(keys)
            if keys not in self.join_featurizer:
                print("join_str: {} not found in featurizer".format(join_str))
                # return jfeats
            else:
                jfeats[self.join_featurizer[keys]] = 1.00

        if use_stats:
            jstart,_ = self.featurizer_type_idxs["join_stats"]

            if not self.feat_separate_alias:
                join_str = ''.join([ck for ck in join_str if not ck.isdigit()])

            join_keys = join_str.split("=")
            ordered_join_keys = [None]*2
            found_primary_key = False
            join_keys.sort()
            for ji, joinkey in enumerate(join_keys):
                if joinkey in self.primary_join_keys:
                    ordered_join_keys[0] = joinkey
                    found_primary_key = True
                    other_key = None
                    for joinkey2 in join_keys:
                        # joinkey2 = joinkey2.strip()
                        if joinkey2 != joinkey:
                            other_key = joinkey2
                    assert other_key is not None
                    ordered_join_keys[1] = other_key
                    break

                ordered_join_keys[ji] = joinkey

            if found_primary_key:
                jfeats[jstart+0] = 1.0
            else:
                jfeats[jstart+1] = 1.0

            key1 = ordered_join_keys[0]
            key2 = ordered_join_keys[1]
            jk1 = ''.join([ck for ck in key1 if not ck.isdigit()])
            jk2 = ''.join([ck for ck in key2 if not ck.isdigit()])

            if jk1 == jk2:
                jfeats[jstart+2] = 1.0
            else:
                jfeats[jstart+2] = 0.0

            jfeats[jstart+3] = float(self.join_key_stats[key1]["distinct"]) \
                    / self.join_key_stats[key2]["distinct"]

            for ji, joinkey in enumerate(ordered_join_keys):
                sidx = jstart + 4 + ji*len(self.join_key_stat_names)
                statdata = self.join_key_stats[joinkey]
                for si, statname in enumerate(self.join_key_stat_names):
                    val = statdata[statname]
                    statmean, statstd = self.join_key_normalizers[statname]
                    jfeats[sidx+si] = (val-statmean) / statstd

        return jfeats

        # if self.join_features == "onehot_tables":
            # jfeats  = np.zeros(self.join_features_len)
            # keys = join_str.split("=")
            # for key in keys:
                # curalias = key[0:key.find(".")]
                # curtab = self.aliases[curalias]
                # tidx = self.table_featurizer[curtab]
                # jfeats[tidx] = 1.0
            # return jfeats

        # elif self.join_features == "onehot":
            # jfeats  = np.zeros(self.join_features_len)
            # if not self.feat_separate_alias:
                # join_str = ''.join([ck for ck in join_str if not ck.isdigit()])

            # keys = join_str.split("=")
            # keys.sort()
            # keys = ",".join(keys)
            # if keys not in self.join_featurizer:
                # print("join_str: {} not found in featurizer".format(join_str))
                # return jfeats
            # jfeats[self.join_featurizer[keys]] = 1.00
            # return jfeats

        # elif self.join_features == "stats":
            # jfeats = np.zeros(self.join_features_len)
            # join_keys = join_str.split("=")
            # ordered_join_keys = [None]*2

            # found_primary_key = False
            # join_keys.sort()
            # for ji, joinkey in enumerate(join_keys):
                # if joinkey in self.primary_join_keys:
                    # ordered_join_keys[0] = joinkey
                    # found_primary_key = True
                    # other_key = None
                    # for joinkey2 in join_keys:
                        # # joinkey2 = joinkey2.strip()
                        # if joinkey2 != joinkey:
                            # other_key = joinkey2
                    # assert other_key is not None
                    # ordered_join_keys[1] = other_key
                    # break

                # ordered_join_keys[ji] = joinkey

            # if found_primary_key:
                # jfeats[0] = 1.0
            # else:
                # jfeats[1] = 1.0

            # key1 = ordered_join_keys[0]
            # key2 = ordered_join_keys[1]
            # jk1 = ''.join([ck for ck in key1 if not ck.isdigit()])
            # jk2 = ''.join([ck for ck in key2 if not ck.isdigit()])

            # if jk1 == jk2:
                # jfeats[2] = 1.0
            # else:
                # jfeats[2] = 0.0

            # jfeats[3] = float(self.join_key_stats[key1]["distinct"]) \
                    # / self.join_key_stats[key2]["distinct"]

            # for ji, joinkey in enumerate(ordered_join_keys):
                # sidx = 4 + ji*len(self.join_key_stat_names)
                # statdata = self.join_key_stats[joinkey]
                # for si, statname in enumerate(self.join_key_stat_names):
                    # val = statdata[statname]
                    # statmean, statstd = self.join_key_normalizers[statname]
                    # jfeats[sidx+si] = (val-statmean) / statstd

            # return jfeats
        # else:
            # assert False

    def _update_set_column_features(self, col, pfeats):
        assert col in self.column_stats
        use_onehot = "onehot" in self.set_column_feature
        use_stats = "stats" in self.set_column_feature

        if use_onehot:
            feat_start,_ = self.featurizer_type_idxs["col_onehot"]
            # which column does the current feature belong to
            cidx = self.columns_onehot_idx[col]
            pfeats[feat_start + cidx] = 1.0

        if use_stats:
            feat_start,_ = self.featurizer_type_idxs["col_stats"]
            statdata = self.join_key_stats[col]
            for si, statname in enumerate(self.join_key_stat_names):
                val = statdata[statname]
                statmean, statstd = self.join_key_normalizers[statname]
                pfeats[feat_start+si] = (val-statmean) / statstd

    def _handle_single_col(self, col, val,
            alias_est, subp_est,
            cmp_op, continuous):

        ret_feats = []
        feat_idx_start = 0

        pfeats = np.zeros(self.max_pred_len)
        self._update_set_column_features(col, pfeats)

        # feat_idx_start += self.set_column_features_len

        # set comparison operator 1-hot value, same for all types
        cmp_start,_ = self.featurizer_type_idxs["cmp_op"]
        cmp_idx = self.cmp_ops_onehot[cmp_op]
        pfeats[cmp_start + cmp_idx] = 1.00

        col_info = self.column_stats[col]
        toaddpfeats = True
        if continuous:
            cstart,_ = self.featurizer_type_idxs["constant_continuous"]
            self._handle_continuous_feature(pfeats, cstart, col, val)
            ret_feats.append(pfeats)
        else:
            if "like" in cmp_op:
                lstart,_ = self.featurizer_type_idxs["constant_like"]
                self._handle_ilike_feature(pfeats,
                        lstart, col, val)
            else:
                # look at _handle_ilike_feature to know how its used
                # pred_idx_start += self.max_like_featurizing_buckets + 2
                dstart,_ = self.featurizer_type_idxs["constant_discrete"]

                if self.embedding_fn is None \
                        or cmp_op == "eq":
                    self._handle_categorical_feature(pfeats,
                            dstart, col, val)
                else:
                    toaddpfeats = False
                    # now do embeddings for each value
                    assert isinstance(val, list)
                    # node_key = tuple([alias])
                    # alias_est = self._get_pg_est(subsetgraph.nodes()[node_key])
                    curfeats = []
                    for word in val:
                        pf2 = copy.deepcopy(pfeats)
                        self._handle_categorical_feature(pf2,
                                dstart, col, [word])
                        curfeats.append(pf2)
                        assert pf2[-1] == 0.0
                        assert pf2[-2] == 0.0
                        pf2[-2] = alias_est
                        pf2[-1] = subp_est

                    if self.embedding_pooling == "sum":
                        pooled_feats = np.sum(np.array(curfeats), axis=0)
                        ret_feats.append(pooled_feats)
                    else:
                        ret_feats += curfeats

        # add the appropriate postgresql estimate for this table in the
        # subplan; Note that the last elements are reserved for the
        # heuristic estimates for both continuous / categorical
        # features
        if self.heuristic_features \
                and toaddpfeats:
            assert pfeats[-1] == 0.0
            assert pfeats[-2] == 0.0
            pfeats[-2] = alias_est
            pfeats[-1] = subp_est
            # test:
            hstart,_ = self.featurizer_type_idxs["heuristic_ests"]
            assert pfeats[hstart] == alias_est
            assert pfeats[hstart+1] == subp_est

        if toaddpfeats:
            ret_feats.append(pfeats)

        return ret_feats

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
                    # assert False
                    continue
                # Note: same table might be set to 1.0 twice, in case of aliases
                tfeats[self.table_featurizer[table]] = 1.00
                alltablefeats.append(tfeats)

        featdict["table"] = alltablefeats

        alljoinfeats = []
        if self.join_features:
            seenjoins = set()
            for alias1 in subplan:
                for alias2 in subplan:
                    ekey = (alias1, alias2)
                    if ekey in joingraph.edges():
                        join_str = joingraph.edges()[ekey]["join_condition"]
                        if join_str in seenjoins:
                            continue
                        seenjoins.add(join_str)
                        jfeats = self._handle_join_features(join_str)
                        alljoinfeats.append(jfeats)

            if len(alljoinfeats) == 0:
                alljoinfeats.append(np.zeros(self.join_features_len))

        featdict["join"] = alljoinfeats

        allpredfeats = []
        for alias in subplan:
            if not self.pred_features:
                continue
            aliasinfo = joingraph.nodes()[alias]
            if len(aliasinfo["pred_cols"]) == 0:
                continue

            node_key = tuple([alias])
            alias_est = self._get_pg_est(subsetgraph.nodes()[node_key])
            subp_est = self._get_pg_est(subsetgraph.nodes()[subplan])
            # print("DEBUGGING subplan est = 0")
            # subp_est = 0.0

            seencols = set()
            for ci, col in enumerate(aliasinfo["pred_cols"]):
                # we should have updated self.column_stats etc. to be appropriately
                # updated
                if not self.feat_separate_alias:
                    col = ''.join([ck for ck in col if not ck.isdigit()])

                if col not in self.column_stats:
                    # print("col: {} not found in column stats".format(col))
                    # assert False
                    continue

                allvals = aliasinfo["pred_vals"][ci]
                if isinstance(allvals, dict):
                    allvals = allvals["literal"]
                cmp_op = aliasinfo["pred_types"][ci]
                continuous = self.column_stats[col]["continuous"]

                if continuous and col in seencols:
                    continue
                seencols.add(col)
                pfeats = self._handle_single_col(col,allvals,
                        alias_est, subp_est,
                        cmp_op,
                        continuous)
                allpredfeats += pfeats

        ## FIXME: need to test this
        # using mcvs for implied preds
        for alias in subplan:
            if not self.implied_pred_features:
                continue

            aliasinfo = joingraph.nodes()[alias]
            if not ("implied_pred_cols" in aliasinfo and \
                    len(aliasinfo["implied_pred_cols"]) > 0):
                continue

            node_key = tuple([alias])

            alias_est = 0.0
            subp_est = self._get_pg_est(subsetgraph.nodes()[subplan])

            for ci, col in enumerate(aliasinfo["implied_pred_cols"]):
                implied_pred_from = aliasinfo["implied_pred_from"][ci]
                implied_pred_alias = implied_pred_from[0:implied_pred_from.find(".")]

                # implied_pred only matters if this table also in the join
                if implied_pred_alias not in subplan:
                    continue

                cmp_op = "in"
                continuous = False

                allvals = aliasinfo["implied_pred_vals"][ci]
                if isinstance(allvals, dict):
                    allvals = allvals["literal"]
                pfeats = self._handle_single_col(col,allvals,
                        alias_est, subp_est,
                        cmp_op,
                        continuous)
                allpredfeats += pfeats

        if len(allpredfeats) == 0:
            allpredfeats.append(np.zeros(self.max_pred_len))

        assert len(allpredfeats) <= self.max_pred_vals

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
                    # print("table: {} not found in featurizer".format(table))
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
                            # print("join_str: {} not found in featurizer".format(join_str))
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
                    # print("col: {} not found in featurizer".format(col))
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
        if "actual" in cardinfo[self.ckey]:
            true_val = cardinfo[self.ckey]["actual"]
        else:
            # e.g., in MLSys competition where we dont want to publish true
            # values
            true_val = 1.0

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

    def unnormalize(self, y, total):
        if self.ynormalization == "log":
            est_card = np.exp((y + \
                self.min_val)*(self.max_val-self.min_val))
        elif self.ynormalization == "selectivity":
            est_card = y*total
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

    def _update_mcvs(self, column):
        # TODO: just need table+column w/o alias here
        if column in self.mcvs:
            return

        table = column[0:column.find(".")]
        attr_name = column[column.find(".")+1:]
        if table in self.aliases:
            table_real_name = self.aliases[table]

        mcv_cmd = MCV_TEMPLATE.format(TABLE = table_real_name,
                                      COL = attr_name)
        mcvres = self.execute(mcv_cmd)
        if len(mcvres) == 0:
            return
        if len(mcvres[0]) == 0:
            return
        if mcvres[0][0] is None:
            return

        mcvs = mcvres[0][0].replace("{","")
        mcvs = mcvs.replace("}","")
        mcvs = mcvs.split(",")
        mcvfreqs = mcvres[0][1]

        if not len(mcvs) == len(mcvfreqs):
            newmcvs = []
            cur_discont_idx = 0
            for mi, mval in enumerate(mcvs):
                if '"' == mval[0] and not '"' == mval[-1]:
                    newval = ""
                    for mi2,mval2 in enumerate(mcvs):
                        if mi2 < mi:
                            continue
                        newval += mval2
                        if '"' == mval2[-1]:
                            cur_discont_idx = mi2
                            break
                        newval += ","
                else:
                    if mi < cur_discont_idx:
                        continue
                    newmcvs.append(mval)
            mcvs = newmcvs
            assert len(mcvs) == len(mcvfreqs)

        total_count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE =
                table_real_name)
        count = int(self.execute(total_count_query)[0][0])

        mcvdict = {}
        for mi, mcv in enumerate(mcvs):
            mcvdict[mcv] = count*mcvfreqs[mi]

        self.mcvs[column] = mcvdict

    def _update_stats(self, qrep):
        '''
        '''
        if qrep["template_name"] not in self.templates:
            self.templates.append(qrep["template_name"])

        node_data = qrep["join_graph"].nodes(data=True)

        num_tables = len(node_data)
        if num_tables > self.max_tables:
            self.max_tables = num_tables

        cur_columns = []
        for node, info in qrep["join_graph"].nodes(data=True):
            if node not in self.aliases:
                self.aliases[node] = info["real_name"]
                self.tables.add(info["real_name"])

            for col in info["pred_cols"]:
                cur_columns.append(col)

            if "implied_pred_cols" in info:
                for col in info["implied_pred_cols"]:
                    cur_columns.append(col)

        joins = extract_join_clause(qrep["sql"])

        for join in joins:
            join = join.replace(" ", "")
            keys = join.split("=")
            keys.sort()
            keystr = ",".join(keys)
            for jkey in keys:
                if jkey in self.join_key_stats:
                    continue
                print("collecting join stats for: {}".format(jkey))
                self.join_key_stats[jkey] = {}
                curalias = jkey[0:jkey.find(".")]
                curcol = jkey[jkey.find(".")+1:]
                if curcol == "id":
                    self.primary_join_keys.add(jkey)

                curtab = self.aliases[curalias]
                # print("skipping join stats for: ", jkey)
                # continue
                for si,tmp in enumerate(self.join_key_stat_tmps):
                    sname = self.join_key_stat_names[si]
                    execcmd = tmp.format(TABLE=curtab,
                                         COL=curcol)
                    try:
                        val = float(self.execute(execcmd)[0][0])
                    except:
                        val = 0.0
                    self.join_key_stats[jkey][sname] = val

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

            # FIXME: reusing join key code here
            jkey = column
            self.join_key_stats[jkey] = {}
            curalias = jkey[0:jkey.find(".")]
            curcol = jkey[jkey.find(".")+1:]
            curtab = self.aliases[curalias]
            for si,tmp in enumerate(self.join_key_stat_tmps):
                sname = self.join_key_stat_names[si]
                execcmd = tmp.format(TABLE=curtab,
                                     COL=curcol)
                try:
                    val = float(self.execute(execcmd)[0][0])
                except:
                    val = 0.0
                self.join_key_stats[jkey][sname] = val

            updated_cols.append(column)
            self._update_mcvs(column)

            table = column[0:column.find(".")]
            column_stats = {}
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
            minval = self.execute(min_query)[0][0]
            maxval = self.execute(max_query)[0][0]

            column_stats[column]["num_values"] = \
                    self.execute(unique_count_query)[0][0]
            column_stats[column]["total_values"] = \
                    self.execute(total_count_query)[0][0]

            column_stats[column]["max_value"] = maxval
            column_stats[column]["min_value"] = minval

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
