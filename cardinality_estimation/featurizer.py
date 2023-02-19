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
import torch

NEW_JOIN_TABLE_TEMPLATE = "{TABLE}_{JOINKEY}_{SS}{NUM}"

JOIN_MAP_IMDB = {}
JOIN_MAP_IMDB["title.id"] = "movie_id"
JOIN_MAP_IMDB["movie_info.movie_id"] = "movie_id"
JOIN_MAP_IMDB["cast_info.movie_id"] = "movie_id"
JOIN_MAP_IMDB["movie_keyword.movie_id"] = "movie_id"
JOIN_MAP_IMDB["movie_companies.movie_id"] = "movie_id"
JOIN_MAP_IMDB["movie_link.movie_id"] = "movie_id"
JOIN_MAP_IMDB["movie_info_idx.movie_id"] = "movie_id"
JOIN_MAP_IMDB["movie_link.linked_movie_id"] = "movie_id"
## TODO: handle it so same columns map to same table+col
# JOIN_MAP_IMDB["miidx.movie_id"] = "movie_id"
JOIN_MAP_IMDB["aka_title.movie_id"] = "movie_id"
JOIN_MAP_IMDB["complete_cast.movie_id"] = "movie_id"

JOIN_MAP_IMDB["movie_keyword.keyword_id"] = "keyword"
JOIN_MAP_IMDB["keyword.id"] = "keyword"

JOIN_MAP_IMDB["name.id"] = "person_id"
JOIN_MAP_IMDB["aka_name.id"] = "person_id"
JOIN_MAP_IMDB["person_info.person_id"] = "person_id"
JOIN_MAP_IMDB["cast_info.person_id"] = "person_id"
JOIN_MAP_IMDB["aka_name.person_id"] = "person_id"
# TODO: handle cases
# JOIN_MAP_IMDB["a.person_id"] = "person_id"

JOIN_MAP_IMDB["title.kind_id"] = "kind_id"
JOIN_MAP_IMDB["aka_title.kind_id"] = "kind_id"
JOIN_MAP_IMDB["kind_type.id"] = "kind_id"

JOIN_MAP_IMDB["cast_info.role_id"] = "role_id"
JOIN_MAP_IMDB["role_type.id"] = "role_id"

JOIN_MAP_IMDB["cast_info.person_role_id"] = "char_id"
JOIN_MAP_IMDB["char_name.id"] = "char_id"

JOIN_MAP_IMDB["movie_info.info_type_id"] = "info_id"
JOIN_MAP_IMDB["movie_info_idx.info_type_id"] = "info_id"
# JOIN_MAP_IMDB["mi_idx.info_type_id"] = "info_id"
# JOIN_MAP_IMDB["miidx.info_type_id"] = "info_id"

JOIN_MAP_IMDB["person_info.info_type_id"] = "info_id"
JOIN_MAP_IMDB["info_type.id"] = "info_id"

JOIN_MAP_IMDB["movie_companies.company_type_id"] = "company_type"
JOIN_MAP_IMDB["company_type.id"] = "company_type"

JOIN_MAP_IMDB["movie_companies.company_id"] = "company_id"
JOIN_MAP_IMDB["company_name.id"] = "company_id"

JOIN_MAP_IMDB["movie_link.link_type_id"] = "link_id"
JOIN_MAP_IMDB["link_type.id"] = "link_id"

JOIN_MAP_IMDB["complete_cast.status_id"] = "subject"
JOIN_MAP_IMDB["complete_cast.subject_id"] = "subject"
JOIN_MAP_IMDB["comp_cast_type.id"] = "subject"

JOIN_MAP_STATS = {}
JOIN_MAP_STATS["badges.id"] = "badge_id"
JOIN_MAP_STATS["badges.userid"] = "user_id"

JOIN_MAP_STATS["comments.userid"] = "user_id"
JOIN_MAP_STATS["comments.postid"] = "post_id"
JOIN_MAP_STATS["comments.id"] = "comment_id"

JOIN_MAP_STATS["posthistory.id"] = "history_id"
JOIN_MAP_STATS["posthistory.postid"] = "post_id"
JOIN_MAP_STATS["posthistory.userid"] = "user_id"

JOIN_MAP_STATS["postlinks.id"] = "link_id"
JOIN_MAP_STATS["postlinks.postid"] = "post_id"
JOIN_MAP_STATS["postlinks.relatedpostid"] = "post_id"
JOIN_MAP_STATS["postlinks.linktypeid"] = "link_type_id"

JOIN_MAP_STATS["posts.id"] = "post_id"
JOIN_MAP_STATS["posts.posttypeid"] = "post_type_id"
JOIN_MAP_STATS["posts.owneruserid"] = "user_id"
JOIN_MAP_STATS["posts.lasteditoruserid"] = "user_id"

JOIN_MAP_STATS["tags.id"] = "tag_id"
# JOIN_MAP_STATS["tags.excerptpostid"] = "post_id"

JOIN_MAP_STATS["users.id"] = "user_id"

JOIN_MAP_STATS["votes.id"] = "vote_id"
JOIN_MAP_STATS["votes.postid"] = "post_id"
JOIN_MAP_STATS["votes.userid"] = "user_id"

JOIN_MAP_STATS["badges.Id"] = "badge_id"
JOIN_MAP_STATS["badges.UserId"] = "user_id"

# JOIN_MAP_STATS["comments.userid"] = "user_id"
# JOIN_MAP_STATS["comments.postid"] = "post_id"

JOIN_MAP_STATS["comments.UserId"] = "user_id"
JOIN_MAP_STATS["comments.PostId"] = "post_id"

JOIN_MAP_STATS["comments.id"] = "comment_id"

# JOIN_MAP_STATS["posthistory.id"] = "history_id"
# JOIN_MAP_STATS["posthistory.postid"] = "post_id"
# JOIN_MAP_STATS["posthistory.userid"] = "user_id"

JOIN_MAP_STATS["postHistory.Id"] = "history_id"
JOIN_MAP_STATS["postHistory.PostId"] = "post_id"
JOIN_MAP_STATS["postHistory.UserId"] = "user_id"

# JOIN_MAP_STATS["postlinks.id"] = "link_id"
# JOIN_MAP_STATS["postlinks.postid"] = "post_id"
# JOIN_MAP_STATS["postlinks.relatedpostid"] = "post_id"
# JOIN_MAP_STATS["postlinks.linktypeid"] = "link_type_id"

JOIN_MAP_STATS["postLinks.Id"] = "link_id"
JOIN_MAP_STATS["postLinks.PostId"] = "post_id"
JOIN_MAP_STATS["postLinks.RelatedPostId"] = "post_id"
JOIN_MAP_STATS["postLinks.LinkTypeId"] = "link_type_id"

# JOIN_MAP_STATS["posts.id"] = "post_id"
# JOIN_MAP_STATS["posts.posttypeid"] = "post_type_id"
# JOIN_MAP_STATS["posts.owneruserid"] = "user_id"
# JOIN_MAP_STATS["posts.lasteditoruserid"] = "user_id"

JOIN_MAP_STATS["posts.Id"] = "post_id"
JOIN_MAP_STATS["posts.PostTypeId"] = "post_type_id"
JOIN_MAP_STATS["posts.OwnerUserId"] = "user_id"
JOIN_MAP_STATS["posts.LastEditorUserId"] = "user_id"


# JOIN_MAP_STATS["tags.id"] = "tag_id"
# JOIN_MAP_STATS["tags.excerptpostid"] = "post_id"
JOIN_MAP_STATS["tags.Id"] = "tag_id"
JOIN_MAP_STATS["tags.ExcerptPostId"] = "post_id"

# JOIN_MAP_STATS["users.id"] = "user_id"
JOIN_MAP_STATS["users.Id"] = "user_id"

# JOIN_MAP_STATS["votes.id"] = "vote_id"
# JOIN_MAP_STATS["votes.postid"] = "post_id"
# JOIN_MAP_STATS["votes.userid"] = "user_id"
JOIN_MAP_STATS["votes.Id"] = "vote_id"

JOIN_MAP_STATS["votes.PostId"] = "post_id"
JOIN_MAP_STATS["votes.UserId"] = "user_id"

JOIN_MAP_ERGAST = {}
JOIN_REAL_MAP = {}
JOIN_REAL_MAP["result_id"] = "results.resultId"
JOIN_MAP_ERGAST["results.resultId"] = "result_id"

JOIN_REAL_MAP["driver_id"] = "drivers.driverId"
JOIN_MAP_ERGAST["drivers.driverId"] = "driver_id"
JOIN_MAP_ERGAST["results.driverId"] = "driver_id"
JOIN_MAP_ERGAST["lapTimes.driverId"] = "driver_id"
JOIN_MAP_ERGAST["qualifying.driverId"] = "driver_id"
JOIN_MAP_ERGAST["pitStops.driverId"] = "driver_id"
JOIN_MAP_ERGAST["driverStandings.driverId"] = "driver_id"

JOIN_REAL_MAP["race_id"] = "races.raceId"
JOIN_MAP_ERGAST["races.raceId"] = "race_id"
JOIN_MAP_ERGAST["results.raceId"] = "race_id"
JOIN_MAP_ERGAST["lapTimes.raceId"] = "race_id"
JOIN_MAP_ERGAST["qualifying.raceId"] = "race_id"
JOIN_MAP_ERGAST["constructorResults.raceId"] = "race_id"
JOIN_MAP_ERGAST["constructorStandings.raceId"] = "race_id"
JOIN_MAP_ERGAST["pitStops.raceId"] = "race_id"
JOIN_MAP_ERGAST["driverStandings.raceId"] = "race_id"

JOIN_REAL_MAP["constructor_id"] = "constructors.constructorId"
JOIN_MAP_ERGAST["constructors.constructorId"] = "constructor_id"
JOIN_MAP_ERGAST["results.constructorId"] = "constructor_id"
JOIN_MAP_ERGAST["constructorStandings.constructorId"] = "constructor_id"
JOIN_MAP_ERGAST["constructorResults.constructorId"] = "constructor_id"
JOIN_MAP_ERGAST["qualifying.constructorId"] = "constructor_id"

JOIN_REAL_MAP["circuit_id"] = "circuits.circuitId"
JOIN_MAP_ERGAST["circuits.circuitId"] = "circuit_id"
JOIN_MAP_ERGAST["races.circuitId"] = "circuit_id"

JOIN_REAL_MAP["status_id"] = "status.statusId"
JOIN_MAP_ERGAST["status.statusId"] = "status_id"
JOIN_MAP_ERGAST["results.statusId"] = "status_id"

JOIN_MAP_SYNTH = {}
JOIN_MAP_SYNTH["synth_primary.id"] = "pid"
JOIN_MAP_SYNTH["synth_foreign.tid"] = "pid"

JOIN_KEY_MAX_TMP = """SELECT COUNT(*), {COL} FROM {TABLE} GROUP BY {COL} ORDER BY COUNT(*) DESC LIMIT 1"""
JOIN_KEY_MIN_TMP = """SELECT COUNT(*), {COL} FROM {TABLE} GROUP BY {COL} ORDER BY COUNT(*) ASC LIMIT 1"""
JOIN_KEY_AVG_TMP = """SELECT AVG(count) FROM (SELECT COUNT(*) AS count, {COL} FROM {TABLE} GROUP BY {COL} ORDER BY COUNT(*)) AS tmp"""
JOIN_KEY_VAR_TMP = """SELECT VARIANCE(count) FROM (SELECT COUNT(*) AS count, {COL} FROM {TABLE} GROUP BY {COL} ORDER BY COUNT(*)) AS tmp"""
JOIN_KEY_COUNT_TMP = """SELECT COUNT({COL}) FROM {TABLE}"""
JOIN_KEY_DISTINCT_TMP = """SELECT COUNT(DISTINCT {COL}) FROM {TABLE}"""

# JOIN_KEY_MAX_TMP = """SELECT COUNT(*), "{COL}" FROM "{TABLE}" GROUP BY "{COL}" ORDER BY COUNT(*) DESC LIMIT 1"""
# JOIN_KEY_MIN_TMP = """SELECT COUNT(*), "{COL}" FROM "{TABLE}" GROUP BY "{COL}" ORDER BY COUNT(*) ASC LIMIT 1"""
# JOIN_KEY_AVG_TMP = """SELECT AVG(count) FROM (SELECT COUNT(*) AS count, "{COL}" FROM "{TABLE}" GROUP BY "{COL}" ORDER BY COUNT(*)) AS tmp"""
# JOIN_KEY_VAR_TMP = """SELECT VARIANCE(count) FROM (SELECT COUNT(*) AS count, "{COL}" FROM "{TABLE}" GROUP BY "{COL}" ORDER BY COUNT(*)) AS tmp"""
# JOIN_KEY_COUNT_TMP = """SELECT COUNT("{COL}") FROM "{TABLE}" """
# JOIN_KEY_DISTINCT_TMP = """SELECT COUNT(DISTINCT "{COL}") FROM "{TABLE}" """

# TODO:
NULL_FRAC_TMP = """SELECT null_frac FROM pg_stats WHERE tablename='{TABLE}' AND attname = '{COL}'"""

CREATE_TABLE_TEMPLATE = "CREATE TABLE {name} (id SERIAL, {columns})"
INSERT_TEMPLATE = "INSERT INTO {name} ({columns}) VALUES %s"

NTILE_CLAUSE = "ntile({BINS}) OVER (ORDER BY {COLUMN}) AS {ALIAS}"
GROUPBY_TEMPLATE = "SELECT {COLS}, COUNT(*) FROM {FROM_CLAUSE} GROUP BY {COLS}"
# COUNT_SIZE_TEMPLATE = """SELECT COUNT(*) FROM "{FROM_CLAUSE}" """
COUNT_SIZE_TEMPLATE = "SELECT COUNT(*) FROM {FROM_CLAUSE}"

SELECT_ALL_COL_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL"
# ALIAS_FORMAT = "{TABLE} AS {ALIAS}"
MIN_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL ORDER BY {COL} ASC LIMIT 1"
MAX_TEMPLATE = "SELECT {COL} FROM {TABLE} WHERE {COL} IS NOT NULL ORDER BY {COL} DESC LIMIT 1"

# SELECT_ALL_COL_TEMPLATE = """ SELECT "{COL}" FROM "{TABLE}" WHERE "{COL}" IS NOT NULL """
ALIAS_FORMAT = """ "{TABLE}" AS {ALIAS} """
# MIN_TEMPLATE = """ SELECT {COL} FROM "{TABLE}" WHERE "{COL}" IS NOT NULL ORDER BY {COL} ASC LIMIT 1 """
# MAX_TEMPLATE = """ SELECT "{COL}" FROM "{TABLE}" WHERE "{COL}" IS NOT NULL ORDER BY "{COL}" DESC LIMIT 1 """

UNIQUE_VALS_TEMPLATE = """SELECT DISTINCT {COL} FROM {FROM_CLAUSE}"""
UNIQUE_COUNT_TEMPLATE = """SELECT COUNT(*) FROM (SELECT DISTINCT {COL} from {FROM_CLAUSE}) AS t"""

# UNIQUE_VALS_TEMPLATE = """SELECT DISTINCT "{COL}" FROM {FROM_CLAUSE}"""
# UNIQUE_COUNT_TEMPLATE = """SELECT COUNT(*) FROM (SELECT DISTINCT "{COL}" from {FROM_CLAUSE}) AS t"""

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

MAX_CEB_IMDB = 23795596119
MAX_JOB = 5607347034
TIMEOUT_CARD = 150001000000

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

class Featurizer():
    def __init__(self, **kwargs):
        '''
        '''
        # self.user = user
        # self.pwd = pwd
        # self.db_host = db_host
        # self.port = port
        # self.db_name = db_name
        for k, val in kwargs.items():
            self.__setattr__(k, val)

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
        # self.join_key_stat_tmps = [NULL_FRAC_TMP, JOIN_KEY_COUNT_TMP,
                # JOIN_KEY_DISTINCT_TMP, JOIN_KEY_AVG_TMP, JOIN_KEY_VAR_TMP,
                # JOIN_KEY_MAX_TMP, JOIN_KEY_MIN_TMP]

        self.join_key_stat_tmps = []

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
        # self.ilike_bigrams = True
        # self.ilike_trigrams = True

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

    def update_workload_means(self, qreps):
        '''
        this is called for a particular set of samples; calculate mean / std of
        cardinalities so we can scale estimates based on that.
        TODO: handle pg ests and true cards differently?
        '''
        if self.clamp_timeouts:
            y0 = []
            # get max value, so we can replace timeout values with it
            for qrep in qreps:
                jg = qrep["join_graph"]
                for node,data in qrep["subset_graph"].nodes(data=True):
                    # if max_num_tables != -1 and len(node) > max_num_tables:
                        # continue
                    actual = data[self.ckey]["actual"]
                    if actual >= TIMEOUT_CARD:
                        continue
                    y0.append(actual)
            maxval = np.max(y0)

        y = []
        yhats = []

        for qrep in qreps:
            jg = qrep["join_graph"]
            for node,data in qrep["subset_graph"].nodes(data=True):
                actual = data[self.ckey]["actual"]
                pg = data[self.ckey]["expected"]
                if self.clamp_timeouts:
                    if actual >= TIMEOUT_CARD:
                        y.append(maxval)
                        yhats.append(maxval)
                    else:
                        y.append(actual)
                        yhats.append(pg)
                else:
                    y.append(actual)
                    yhats.append(pg)

        y = np.array(y)
        yhats = np.array(yhats)

        if "log" in self.ynormalization:
            y = np.log(y)
            yhats = np.log(yhats)

        self.meany = np.mean(y)
        self.stdy = np.std(y)

        print("mean y: {}, std y: {}".format(self.meany, self.stdy))
        print("PG Estimates: mean y: {}, std y: {}".format(np.mean(yhats),
            np.std(yhats)))

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

    def join_str_to_real_join(self, joinstr):
        # return joinstr

        join_tabs = joinstr.split("=")
        join_tabs.sort()
        real_join_tabs = []

        for jt in join_tabs:
            jt = jt.replace(" ", "")
            jsplits = jt.split(".")
            tab_alias = jsplits[0]
            if tab_alias not in self.aliases:
                print("tab alias not in self.aliases: ", tab_alias)
                # pdb.set_trace()
            real_jkey = self.aliases[tab_alias] + "." + jsplits[1]
            real_jkey = real_jkey.replace(" ", "")
            real_join_tabs.append(real_jkey)

        return "=".join(real_join_tabs)

    def update_seen_preds(self, qreps):
        # key: column name, val: set() of seen values
        self.seen_preds = {}
        # need separate dictionaries, because like constants / like
        # featurization has very different meaning from categorical
        # featurization
        self.seen_like_preds = {}
        self.seen_joins = set()
        self.seen_tabs = set()
        self.seen_bitmaps = {}

        for qrep in qreps:
            for ekey in qrep["join_graph"].edges():
                join_str = qrep["join_graph"].edges()[ekey]["join_condition"]
                join_str = self.join_str_to_real_join(join_str)
                self.seen_joins.add(join_str)

            if self.sample_bitmap:
                sbitmaps = None
                sbitdir = os.path.join(self.bitmap_dir, qrep["workload"],
                        "sample_bitmap")

                bitmapfn = os.path.join(sbitdir, qrep["name"])

                if not os.path.exists(bitmapfn):
                    print(bitmapfn)
                    # pdb.set_trace()
                    sbitmaps = None
                else:
                    with open(bitmapfn, "rb") as handle:
                        sbitmaps = pickle.load(handle)

            for node, info in qrep["join_graph"].nodes(data=True):
                self.seen_tabs.add(info["real_name"])
                real_name = info["real_name"]
                if real_name == "synth_primary":
                    continue

                if self.sample_bitmap and sbitmaps is not None:
                    if real_name not in self.seen_bitmaps:
                        self.seen_bitmaps[real_name] = set()

                    assert sbitmaps is not None
                    # startidx = len(self.table_featurizer)
                    if (node,) not in sbitmaps:
                        # print(node)
                        # print(real_name)
                        # pdb.set_trace()
                        continue
                    sb = sbitmaps[(node,)]
                    # assert self.sample_bitmap_key in sb
                    if self.sample_bitmap_key not in sb:
                        continue
                    bitmap = sb[self.sample_bitmap_key]
                    for val in bitmap:
                        self.seen_bitmaps[real_name].add(val)

                for ci, col in enumerate(info["pred_cols"]):
                    # cur_columns.append(col)
                    if not self.feat_separate_alias:
                        col = ''.join([ck for ck in col if not ck.isdigit()])

                    if col not in self.seen_preds:
                        self.seen_preds[col] = set()

                    vals = info["pred_vals"][ci]
                    cmp_op = info["pred_types"][ci]
                    if isinstance(vals, list):
                        for val in vals:
                            if isinstance(val, dict):
                                val = val["literal"]
                            self.seen_preds[col].add(val)
                    else:
                        if isinstance(vals, dict):
                            vals = vals["literal"]
                        self.seen_preds[col].add(vals)

                    if "like" in cmp_op:
                        if col not in self.seen_like_preds:
                            self.seen_like_preds[col] = set()
                        assert len(vals) == 1
                        self.seen_like_preds[col].add(vals[0])

        # print(self.seen_bitmaps.keys())
        # pdb.set_trace()

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

    def update_max_sets(self, qreps):
        if "stats" in qreps[0]["template_name"]:
            self.join_col_map = JOIN_MAP_STATS
        elif "ergast" in qreps[0]["workload"]:
            self.join_col_map = JOIN_MAP_ERGAST
        elif "synth" in qreps[0]["workload"]:
            self.join_col_map = JOIN_MAP_SYNTH
        else:
            self.join_col_map = JOIN_MAP_IMDB

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
                if not "pred_vals" in info:
                    continue
                if len(info["pred_vals"]) == 0:
                    continue

                num_preds += len(info["pred_cols"])
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

            # kinda hacky
            if self.max_preds > self.max_pred_vals:
                self.max_pred_vals = self.max_preds

            # length 1 pred sets run into some broadcasting things
            if self.max_preds <= 1:
                self.max_preds = 2

            num_joins = len(qrep["join_graph"].edges())
            if num_joins > self.max_joins:
                self.max_joins = num_joins


        if self.max_joins <= 1:
            self.max_joins = 2

        if self.max_tables <= 1:
            self.max_tables = 2

        if self.join_bitmap:
            self.max_joins = len(set(self.join_col_map.values()))
            self.max_joins += self.max_tables

        print("Max tables:", self.max_tables, ", Max pred vals:", self.max_pred_vals,
                ",Max joins:", self.max_joins)

    def get_cont_pred_kind(info):
        pass

    def update_workload_stats(self, qreps):
        for qrep in qreps:
            cur_columns = []
            for node, info in qrep["join_graph"].nodes(data=True):
                if "pred_types" not in info:
                    print("pred types not in info!")
                    continue

                for i, cmp_op in enumerate(info["pred_types"]):
                    if cmp_op == "lt":
                        if len(info["predicates"]) > i:
                            if ">" in info["predicates"][i]:
                                cmp_op = ">"
                            elif "<" in info["predicates"][i]:
                               cmp_op = "<"

                    self.cmp_ops.add(cmp_op)

                    if "like" in cmp_op:
                        self.regex_cols.add(info["pred_cols"][i])
                        self.regex_templates.add(qrep["template_name"])

                self.tables.add(info["real_name"])

                if node not in self.aliases:
                    self.aliases[node] = info["real_name"]
                    # also without the ints
                    node2 = "".join([n1 for n1 in node if not n1.isdigit()])
                    self.aliases[node2] = info["real_name"]

                for col in info["pred_cols"]:
                    cur_columns.append(col)

            joins = extract_join_clause(qrep["sql"])
            for join_str in joins:
                # get rid of whitespace
                # joinstr = joinstr.replace(" ", "")
                join_str = self.join_str_to_real_join(join_str)
                if not self.feat_separate_alias:
                    join_str = ''.join([ck for ck in join_str if not ck.isdigit()])
                keys = join_str.split("=")
                keys.sort()
                keys = ",".join(keys)
                self.joins.add(keys)

        print("max pred vals: {}".format(self.max_pred_vals))
        print("Seen comparison operators: ", self.cmp_ops)
        print("Tables: ", self.tables)

        # if self.set_column_feature == "debug":
            # print(self.cmp_ops)
            # pdb.set_trace()

    def update_ystats_joinkey(self, qreps,
            max_num_tables=-1):
        y = []
        for qrep in qreps:
            # for node,data in qrep["subset_graph"].nodes(data=True):
            for _,_,data in qrep["subset_graph"].edges(data=True):
                cards = data["join_key_cardinality"]
                for k,v in cards.items():
                    y.append(v["actual"])

        y = np.array(y)

        if np.min(y) == 0:
            y += 1

        if self.ynormalization == "log":
            y = np.log(y)

        self.max_val = np.max(y)
        self.min_val = np.min(y)

        print("min y: {}, max y: {}".format(self.min_val, self.max_val))

    def update_ystats(self, qreps,
            max_num_tables=-1):

        if self.clamp_timeouts:
            y0 = []
            # get max value, so we can replace timeout values with it
            for qrep in qreps:
                jg = qrep["join_graph"]
                for node,data in qrep["subset_graph"].nodes(data=True):
                    if max_num_tables != -1 and len(node) > max_num_tables:
                        continue
                    if "actual" not in data[self.ckey]:
                        continue

                    actual = data[self.ckey]["actual"]
                    if actual >= TIMEOUT_CARD:
                        continue
                    y0.append(actual)

            maxval = np.max(y0)

        y = []

        for qrep in qreps:
            jg = qrep["join_graph"]
            for node,data in qrep["subset_graph"].nodes(data=True):
                if max_num_tables != -1 and len(node) > max_num_tables:
                    continue
                if "actual" not in data[self.ckey]:
                    continue

                actual = data[self.ckey]["actual"]
                if self.clamp_timeouts:
                    if actual >= TIMEOUT_CARD:
                        y.append(maxval)
                    else:
                        y.append(actual)
                else:
                    y.append(actual)

                if self.ynormalization == "selectivity":
                    y[-1] = float(y[-1]) / data[self.ckey]["total"]

        y = np.array(y)

        if np.min(y) == 0:
            y += 1

        if "log" in self.ynormalization:
            y = np.log(y)

        self.max_val = np.max(y)
        self.min_val = np.min(y)

        print("min y: {}, max y: {}".format(self.min_val, self.max_val))

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
            # pdb.set_trace()
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
                    # and "synth" not in self.db_name:
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

        self.num_cols = len(self.column_stats)
        all_cols = list(self.column_stats.keys())
        all_cols.sort()

        self.columns_onehot_idx = {}

        real_col_idxs = {}
        ridx = 0

        for cidx, col_name in enumerate(all_cols):
            col_splits = col_name.split(".")
            col_alias = col_splits[0]
            # we can have new aliases here because column_stats is created on
            # all used imdb columns from before --- not just in the current
            # workload
            if col_alias not in self.aliases:
                continue

            real_col = self.aliases[col_alias] + "." + col_splits[1]

            if real_col in real_col_idxs:
                self.columns_onehot_idx[col_name] = real_col_idxs[real_col]
            else:
                real_col_idxs[real_col] = ridx
                self.columns_onehot_idx[col_name] = ridx
                ridx += 1

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

        ilike_feat_size = self.max_like_featurizing_buckets
        if self.like_char_features:
            ilike_feat_size += 2

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
            numh = 2
            if self.feat_separate_like_ests:
                numh += 2
            self.featurizer_type_idxs["heuristic_ests"] = (pred_len, numh)
            # for pg_est of current table in subplan
            # and for pg est of full subplan/query; repeated in each predicate
            # feature
            pred_len += numh

        self.max_pred_len = pred_len
        self.pred_onehot_mask = np.ones(self.max_pred_len)
        self.pred_onehot_mask_consts = np.ones(self.max_pred_len)

        a,b = self.featurizer_type_idxs["constant_discrete"]
        self.pred_onehot_mask[a:a+b] = 0.0
        self.pred_onehot_mask_consts[a:a+b] = 0.0

        a,b = self.featurizer_type_idxs["constant_like"]
        self.pred_onehot_mask[a:a+b] = 0.0
        self.pred_onehot_mask_consts[a:a+b] = 0.0

        if "col_onehot" in self.featurizer_type_idxs:
            a,b = self.featurizer_type_idxs["col_onehot"]
            self.pred_onehot_mask[a:a+b] = 0.0

        ## TODO: apply dropout to these or not?
        # a,b = self.featurizer_type_idxs["constant_continuous"]
        # self.pred_onehot_mask[a:a+b] = 0.0

        # a,b = self.featurizer_type_idxs["cmp_op"]
        # self.pred_onehot_mask[a:a+b] = 0.0

        ## mapping columns to continuous or not
        for col in col_keys:
            info = self.column_stats[col]
            if is_float(info["min_value"]) and is_float(info["max_value"]) \
                    and "id" not in col:
                    # and "synth" not in self.db_name:
                # then use min-max normalization, no matter what
                # only support range-queries, so lower / and upper predicate
                # continuous = True
                self.column_stats[col]["continuous"] = True
            else:
                self.column_stats[col]["continuous"] = False

    def setup(self,
            bitmap_dir = None,
            join_bitmap_dir = None,
            use_saved_feats = True,
            feat_onlyseen_maxy = False,
            max_num_tables = -1,
            true_base_cards = False,
            loss_func = "mse",
            heuristic_features=True,
            like_char_features=False,
            feat_separate_alias=False,
            feat_separate_like_ests=False,
            separate_ilike_bins=False,
            separate_cont_bins=False,
            # onehot_dropout=False,
            ynormalization="log",
            table_features = True,
            pred_features = True,
            feat_onlyseen_preds = True,
            seen_preds = False,
            set_column_feature = "onehot",
            join_features = "onehot",
            global_features = False,
            embedding_fn = None,
            embedding_pooling = None,
            num_tables_feature=False,
            featurization_type="combined",
            card_type = "subplan",
            clamp_timeouts = 1,
            max_discrete_featurizing_buckets=10,
            max_like_featurizing_buckets=10,
            feat_num_paths= False, feat_flows=False,
            feat_pg_costs = False, feat_tolerance=False,
            feat_pg_path=False,
            global_feat_degrees = False,
            global_feat_tables = False,
            feat_rel_pg_ests=False, feat_join_graph_neighbors=False,
            feat_rel_pg_ests_onehot=False,
            feat_pg_est_one_hot=False,
            feat_mcvs = False,
            implied_pred_features=False,
            cost_model=None,
            sample_bitmap=False,
            join_bitmap = False,
            bitmap_onehotmask=False,
            sample_bitmap_num=1000,
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

            Other features are flags for various minor tweaks / additional
            features. For most use cases, the default values should suffice.

        This generates a featurizer dict:
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
            if "synth" not in self.db_name:
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

            if self.loss_func == "flowloss":
                print("updating global features to include flowloss specific ones")
                self.feat_num_paths= False
                self.feat_flows=False
                self.feat_pg_costs = True
                self.feat_tolerance=False
                self.feat_pg_path=True
                self.feat_rel_pg_ests=False
                self.feat_join_graph_neighbors=True
                self.feat_rel_pg_ests_onehot=True
                self.feat_pg_est_one_hot=True
                self.global_feat_degrees = True

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

        if self.join_bitmap:
            self.sample_bitmap_key = "sb" + str(self.sample_bitmap_num)

        if self.sample_bitmap:
            # bitmap_tables = []
            self.sample_bitmap_key = "sb" + str(self.sample_bitmap_num)
            if self.featurization_type == "set":
                self.table_features_len = len(self.tables) + self.sample_bitmap_num
                self.max_table_feature_len = len(self.tables) + self.sample_bitmap_num
            else:
                self.table_features_len = len(self.tables) + len(self.tables)*self.sample_bitmap_buckets
                self.max_table_feature_len = len(self.tables) + \
                            len(self.tables)*self.sample_bitmap_buckets
        else:

            self.table_features_len = len(self.tables)
            self.max_table_feature_len = len(self.tables)
            if self.table_features_len <= 1:
                self.table_features_len = 2

        # only have onehot encoding for tables
        self.table_onehot_mask = np.zeros(self.table_features_len)

        use_onehot = ("onehot" in self.join_features \
                            or "1" in self.join_features) \
                            and not self.join_bitmap
        use_stats = "stats" in self.join_features and not self.join_bitmap

        ## join features
        if self.join_features == "1":
            # or table one
            self.join_features = "onehot"
        elif self.join_features == "0":
            self.join_features = False


        self.join_features_len = 0

        if self.join_bitmap:
            bitmap_feat_len = 0
            # for #tables in the join
            bitmap_feat_len += self.max_tables
            # for which tables are there
            bitmap_feat_len += len(self.tables)

            if self.featurization_type == "combined":
                bitmap_feat_len = len(set(self.join_col_map.values()))*self.sample_bitmap_buckets

            else:
                # for real col
                bitmap_feat_len += len(set(self.join_col_map.values()))

                if not self.bitmap_onehotmask:
                    self.featurizer_type_idxs["join_onehot"] = (0, bitmap_feat_len)

                bitmap_feat_len += self.sample_bitmap_buckets

            ## includes everything for the onehot-mask
            if self.bitmap_onehotmask:
                self.featurizer_type_idxs["join_onehot"] = (0, bitmap_feat_len)

            self.join_features_len += bitmap_feat_len

        if use_onehot:
            self.join_featurizer = {}

            for i, join in enumerate(sorted(self.joins)):
                self.join_featurizer[join] = i

            self.join_features_len += len(self.joins)
            ## avoiding edge cases with broadcasting length 1 things
            self.join_features_len = max(self.join_features_len, 2)
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

        ## avoids size 1 broadcasting edge cases
        if self.join_features_len <= 1:
            self.join_features_len = 2

        self.join_onehot_mask = np.ones(self.join_features_len)

        if "join_onehot" in self.featurizer_type_idxs:
            a,b = self.featurizer_type_idxs["join_onehot"]
            self.join_onehot_mask[a:b] = 0.0

        ## predicate filter features
        if self.featurization_type == "combined":
            self._init_pred_featurizer_combined()
        elif self.featurization_type == "set":
            self._init_pred_featurizer_set()

        real_join_cols = list(set(self.join_col_map.values()))
        real_join_cols.sort()
        self.real_join_col_mapping = {}
        for rci,rc in enumerate(real_join_cols):
            self.real_join_col_mapping[rc] = rci

        self.PG_EST_BUCKETS = 7
        if self.global_features:
            # num flow features: concat of 1-hot vectors
            self.num_global_features = 0

            if self.card_type == "joinkey":
                self.num_global_features += len(self.real_join_col_mapping)

            if self.global_feat_degrees:
                self.num_global_features += self.max_in_degree+1
                self.num_global_features += self.max_out_degree+1

            if self.global_feat_tables:
                self.num_global_features += len(self.aliases)

            # for heuristic estimate for the node
            self.num_global_features += 1

            # for normalized value of num_paths
            if self.feat_num_paths:
                self.num_global_features += 1
            if self.feat_pg_costs:
                self.num_global_features += 1
            if self.feat_tolerance:
                # 1-hot vector based on dividing/multiplying value by 10...10^4
                self.num_global_features += 4
            if self.feat_flows:
                self.num_global_features += 1

            if self.feat_pg_path:
                self.num_global_features += 1

            if self.feat_rel_pg_ests:
                # current node size est, relative to total cost
                self.num_global_features += 1

                # current node est, relative to all neighbors in the join graph
                # we will hard code the neighbor into a 1-hot vector
                self.num_global_features += len(self.table_featurizer)

            if self.feat_rel_pg_ests_onehot:
                self.num_global_features += self.PG_EST_BUCKETS
                # 2x because it can be smaller or larger
                self.num_global_features += \
                    2*len(self.table_featurizer)*self.PG_EST_BUCKETS

            if self.feat_join_graph_neighbors:
                self.num_global_features += len(self.table_featurizer)

            if self.feat_pg_est_one_hot:
                # upto 10^7
                self.num_global_features += self.PG_EST_BUCKETS

            print("Number of global features: ", self.num_global_features)

    def _handle_continuous_feature(self, pfeats, pred_idx_start,
            col, val):
        '''
        '''
        col_info = self.column_stats[col]
        min_val = float(col_info["min_value"])
        max_val = float(col_info["max_value"])

        if "synth" in self.db_name:
            val = val[0] if val[0] is not None else val[1]
            cur_val = float(val)
            norm_val = (cur_val - min_val) / (max_val - min_val)
            norm_val = max(norm_val, 0.00)
            norm_val = min(norm_val, 1.00)
            pfeats[pred_idx_start] = norm_val
            return

        # assert isinstance(val, list)
        if not isinstance(val, list):
            val = [None, val]

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
            try:
                cur_val = float(v)
                norm_val = (cur_val - min_val) / (max_val - min_val)
                norm_val = max(norm_val, 0.00)
                norm_val = min(norm_val, 1.00)
                pfeats[pred_idx_start+vi] = norm_val
            except:
                pass

    def _handle_categorical_feature(self, pfeats,
            pred_idx_start, col, val):
        '''
        hashed features;
        '''
        col_info = self.column_stats[col]
        if self.featurization_type == "combined":
            num_buckets = min(self.max_discrete_featurizing_buckets,
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

            if not isinstance(val, list):
                val = [val]

            for v in val:
                if self.feat_onlyseen_preds:
                    if isinstance(v, dict):
                        v = v["literal"]
                        # print(v)
                        # pdb.set_trace()
                    if v not in self.seen_preds[col]:
                        continue

                pred_idx = deterministic_hash(str(v)) % num_buckets
                pfeats[pred_idx_start+pred_idx] = 1.00

    def _get_true_est(self, subpinfo):
        true_est = subpinfo[self.ckey]["actual"]
        # note: total is only needed for self.ynormalization == selectivity

        if "total" in subpinfo[self.ckey]:
            total = subpinfo[self.ckey]["total"]
        else:
            total = None
        subp_est = self.normalize_val(true_est,
                total)
        return subp_est

    def _get_pg_est(self, subpinfo):
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

        if self.like_char_features:
            for v in regex_val:
                pred_idx = deterministic_hash(str(v)) % num_buckets
                pfeats[pred_idx_start+pred_idx] = 1.00

            for i,v in enumerate(regex_val):
                if i != len(regex_val)-1:
                    pred_idx = deterministic_hash(v+regex_val[i+1]) % num_buckets
                    pfeats[pred_idx_start+pred_idx] = 1.00

            for i,v in enumerate(regex_val):
                if i < len(regex_val)-2:
                    pred_idx = deterministic_hash(v+regex_val[i+1]+ \
                            regex_val[i+2]) % num_buckets
                    pfeats[pred_idx_start+pred_idx] = 1.00

            pfeats[pred_idx_start + num_buckets] = len(regex_val)

            # regex has num or not feature
            if bool(re.search(r'\d', regex_val)):
                pfeats[pred_idx_start + num_buckets + 1] = 1

    def _find_join_bitmaps(self, alias, join_bitmaps, bitmaps,
            joingraph):
        # find all potential join keys in this table
        ret_bitmaps = {}
        if join_bitmaps is None:
            return ret_bitmaps
        tab = joingraph.nodes()[alias]["real_name"]
        for join_key, join_real in self.join_col_map.items():
            if ".id" in join_key.lower():
                continue
            if "posttypeid" in join_key.lower():
                continue
            join_key_col1 = join_key[join_key.find(".")+1:]
            join_tab = join_key[:join_key.find(".")]

            if tab == join_tab:
                newid = join_key[join_key.find(".")+1:]
                bitmap_key = NEW_JOIN_TABLE_TEMPLATE.format(TABLE=tab,
                                               JOINKEY = join_real,
                                               SS = "sb",
                                               NUM = self.sample_bitmap_num)
                if (alias,) not in join_bitmaps:
                    # print(alias, " not in join bitmaps")
                    continue
                alias_bm = join_bitmaps[(alias,)]
                if bitmap_key not in alias_bm:
                    continue
                jbitmap = set(alias_bm[bitmap_key])
                ret_bitmaps[join_real] = jbitmap

        return ret_bitmaps

    def _handle_join_bitmaps_combined(self, subplan, join_bitmaps,
            bitmaps,
            joingraph):
        '''
        TODO: need to enforce that joins actually there between all tables
        mapping to same join real col: e.g., mi <-> mi; might not have joins.
        '''
        jfeats  = np.zeros(self.join_features_len)
        if bitmaps is None and join_bitmaps is None:
            return jfeats

        # join_features = []
        # start_idx, end_idx = self.featurizer_type_idxs["join_bitmap"]

        real_join_cols = defaultdict(list)
        real_join_tabs = defaultdict(list)

        if len(subplan) == 1:
            return jfeats

        seenjoins = set()
        for alias1 in subplan:
            alias_jbitmaps  = self._find_join_bitmaps(alias1, join_bitmaps,
                                                         bitmaps, joingraph)
            for rcol, rbm in alias_jbitmaps.items():
                real_join_cols[rcol].append(rbm)

            ## maybe this stuff is just not required?
            ## probably need it for the id special case -- can handle that in
            ## _find_join_bitmaps too maybe?
            for alias2 in subplan:
                ekey = (alias1, alias2)
                if ekey not in joingraph.edges():
                    continue
                join_str = joingraph.edges()[ekey]["join_condition"]
                join_str = self.join_str_to_real_join(join_str)
                if join_str in seenjoins:
                    continue

                cols = join_str.split("=")
                for ci, c in enumerate(cols):
                    if c not in self.join_col_map:
                        c = c.replace("\"", "")

                    if c not in self.join_col_map:
                        print("{} still not in JOIN COL MAP".format(c))
                        # pdb.set_trace()
                        continue

                    rcol = self.join_col_map[c]
                    tabname = c[0:c.find(".")]
                    real_join_tabs[rcol].append(tabname)

                    # find its bitmap
                    if self.aliases[alias1] == tabname:
                        curalias = alias1
                    elif self.aliases[alias2] == tabname:
                        curalias = alias2
                    else:
                        assert False

                    if ".id" in c.lower():
                        # sample bitmap
                        if bitmaps is None:
                            continue
                        if (curalias,) not in bitmaps:
                            continue
                        if self.sample_bitmap_key not in bitmaps[(curalias,)]:
                            continue
                        try:
                            sb = bitmaps[(curalias,)][self.sample_bitmap_key]
                            bitmap = set(sb)
                        except Exception as e:
                            print(bitmaps)
                            print(curalias)
                            pdb.set_trace()
                    else:
                        bitmap_key = NEW_JOIN_TABLE_TEMPLATE.format(
                                TABLE=tabname,
                                JOINKEY=rcol,
                                SS="sb",
                                NUM=self.sample_bitmap_num)

                        alias_bm = join_bitmaps[(curalias,)]
                        # TODO: maybe handle this some other way?
                        if bitmap_key not in alias_bm:
                            continue

                        bitmap = set(alias_bm[bitmap_key])
                        print(bitmap)
                        pdb.set_trace()

                    real_join_cols[rcol].append(bitmap)

        cur_idx = 0
        # features = np.zeros(self.join_features_len)

        # num tables
        # alltabs = real_join_tabs[rc]
        # nt_idx = len(alltabs)-1
        # jfeats[cur_idx + nt_idx] = 1.0
        # cur_idx += self.max_tables

        for rc in real_join_cols:
            ## bitmap
            # real column on which we have bitmap
            start_idx = self.real_join_col_mapping[rc]*self.sample_bitmap_buckets
            # features[cur_idx + jcol_idx] = 1.0
            # cur_idx += len(self.real_join_col_mapping)

            # for table in alltabs:
                # if table not in self.table_featurizer:
                    # print("table: {} not found in featurizer".format(table))
                    # pdb.set_trace()
                    # continue
                # # Note: same table might be set to 1.0 twice, in case of aliases
                # jfeats[start_idx + self.table_featurizer[table]] = 1.00
            # cur_idx += len(self.table_featurizer)

            # bitmap intersection
            bitmap_int = set.intersection(*real_join_cols[rc])

            for val in bitmap_int:
                bitmapidx = val % self.sample_bitmap_buckets
                jfeats[start_idx+bitmapidx] = 1.0

        return jfeats

    def _handle_join_bitmaps(self, subplan, join_bitmaps,
            bitmaps,
            joingraph):
        '''
        TODO: need to enforce that joins actually there between all tables
        mapping to same join real col: e.g., mi <-> mi; might not have joins.
        '''
        # assert join_bitmaps is not None
        # assert bitmaps is not None
        if bitmaps is None and join_bitmaps is None:
            return [np.zeros(self.join_features_len)]

        join_features = []

        # start_idx, end_idx = self.featurizer_type_idxs["join_bitmap"]

        real_join_cols = defaultdict(list)
        real_join_tabs = defaultdict(list)

        if len(subplan) == 1:
            return join_features
        if join_bitmaps is None:
            return join_features

        seenjoins = set()
        # print(">>>>>join bitmaps for: ", subplan)

        for alias1 in subplan:
            alias_jbitmaps  = self._find_join_bitmaps(alias1, join_bitmaps,
                                                         bitmaps, joingraph)

            # print(alias_jbitmaps.keys())
            for rcol, rbm in alias_jbitmaps.items():
                if rcol != "result_id":
                    continue
                if "res3" in subplan:
                    continue

                if "1" in alias1:
                    rcol += "1"
                elif "2" in alias1:
                    rcol += "2"

                real_join_cols[rcol].append(rbm)

            # if "res1" in subplan and "res2" in subplan and alias1 == "res1":
                # print(alias1)
                # print(alias_jbitmaps.keys())
                # pdb.set_trace()

            ## maybe this stuff is just not required?
            ## probably need it for the id special case -- can handle that in
            ## _find_join_bitmaps too maybe?
            for alias2 in subplan:
                ekey = (alias1, alias2)
                if ekey not in joingraph.edges():
                    continue

                if "2" in alias1 and "2" in alias2:
                    jgroup = "2"
                elif "1" in alias1 and "1" in alias2:
                    jgroup = "1"
                elif "3" in alias1 and "3" in alias2:
                    jgroup = "3"
                elif "1" in alias1 and "2" in alias2:
                    jgroup = "mix"
                elif "2" in alias1 and "1" in alias2:
                    jgroup = "mix"
                elif "3" in alias1 and "2" in alias2:
                    jgroup = "mix"
                elif "3" in alias1 and "1" in alias2:
                    jgroup = "mix"
                else:
                    jgroup = ""

                join_str = joingraph.edges()[ekey]["join_condition"]

                if alias1 + "," + join_str in seenjoins:
                    continue

                # assert len(seenjoins) == 0
                seenjoins.add(alias1 + "," + join_str)

                # strip alias info
                join_str = self.join_str_to_real_join(join_str)

                cols = join_str.split("=")

                for ci, c in enumerate(cols):
                    if c not in self.join_col_map:
                        c = c.replace("\"", "")

                    if c not in self.join_col_map:
                        print("{} still not in JOIN COL MAP".format(c))
                        continue
                        # pdb.set_trace()

                    rcol = self.join_col_map[c]
                    rcol_orig = self.join_col_map[c]

                    ## special casing driver id join for now
                    # if jgroup != "mix" and rcol != "driver_id":
                        # rcol += jgroup
                    if jgroup != "mix":
                        if not (rcol == "driver_id" and \
                            "d1" in subplan and "d2" in subplan):
                            rcol = rcol + jgroup
                        elif not (rcol == "constructor_id" and \
                            "c1" in subplan and "c2" in subplan):
                            rcol = rcol + jgroup

                    tabname = c[0:c.find(".")]
                    real_join_tabs[rcol].append(tabname)

                    # find its bitmap
                    if self.aliases[alias1] == tabname:
                        curalias = alias1
                    elif self.aliases[alias2] == tabname:
                        # alias2 will be alias1 later
                        continue
                        # curalias = alias2
                    # else:
                        # assert False

                    if ".id" in c.lower():
                        # sample bitmap
                        if bitmaps is None:
                            continue
                        if (curalias,) not in bitmaps:
                            continue
                        if self.sample_bitmap_key not in bitmaps[(curalias,)]:
                            continue
                        try:
                            sb = bitmaps[(curalias,)][self.sample_bitmap_key]
                            bitmap = set(sb)
                        except Exception as e:
                            print(bitmaps)
                            print(curalias)
                            pdb.set_trace()
                    else:
                        bitmap_key = NEW_JOIN_TABLE_TEMPLATE.format(
                                TABLE=tabname,
                                JOINKEY=rcol_orig,
                                SS="sb",
                                NUM=self.sample_bitmap_num)

                        try:
                            alias_bm = join_bitmaps[(curalias,)]
                        except Exception as e:
                            # print("exception in join bitmap")
                            # print(e)
                            continue

                        # print(alias1, alias2)
                        # print(c)
                        # print(join_str)
                        # print(bitmap_key)
                        # print(curalias)
                        # print(curalias)
                        # print(joingraph.nodes()[curalias])
                        # print(set(alias_bm[bitmap_key]))
                        # pdb.set_trace()

                        # TODO: maybe handle this some other way?
                        if bitmap_key not in alias_bm:
                            continue

                        bitmap = set(alias_bm[bitmap_key])

                    real_join_cols[rcol].append(bitmap)

        # print("*****")
        # print(subplan)
        for rc in real_join_cols:
            ## debugging code
            # print(rc)
            # print(len(real_join_cols[rc]))
            # for curvs in real_join_cols[rc]:
                # print(len(curvs))

            bitmap_int = set.intersection(*real_join_cols[rc])

            cur_idx = 0
            features = np.zeros(self.join_features_len)
            # num tables
            alltabs = real_join_tabs[rc]
            nt_idx = len(alltabs)-1
            features[cur_idx + nt_idx] = 1.0
            cur_idx += self.max_tables

            # which tables
            for table in alltabs:
                if table not in self.table_featurizer:
                    print("table: {} not found in featurizer".format(table))
                    pdb.set_trace()
                    continue
                # Note: same table might be set to 1.0 twice, in case of aliases
                features[cur_idx + self.table_featurizer[table]] = 1.00
            cur_idx += len(self.table_featurizer)

            ## bitmap
            # real column on which we have bitmap
            rc_orig = "".join([ch for ch in rc if not ch.isdigit()])
            jcol_idx = self.real_join_col_mapping[rc_orig]
            features[cur_idx + jcol_idx] = 1.0
            cur_idx += len(self.real_join_col_mapping)

            # if rc != "movie_id":
                # print(rc, len(real_join_cols[rc]))
                # pdb.set_trace()

            # bitmap intersection
            bitmap_int = set.intersection(*real_join_cols[rc])
            # init_size = max([len(jc) for jc in real_join_cols[rc]])

            for val in bitmap_int:
                # TODO: check if seen condition or no
                # if self.random_bitmap_idx:
                    # # more robust?
                    # bitmapidx = random.randint(0, self.sample_bitmap_num-1)
                    # features[cur_idx+bitmapidx] = 1.0
                # else:
                bitmapidx = val % self.sample_bitmap_buckets
                features[cur_idx+bitmapidx] = 1.0

            join_features.append(features)

        return join_features

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
                # print(join_str)
                # print(self.join_featurizer)
                # pdb.set_trace()
                return jfeats
                # pass
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

    def _update_set_column_features(self, col, pfeats):
        assert col in self.column_stats
        use_onehot = "onehot" in self.set_column_feature
        use_stats = "stats" in self.set_column_feature

        if use_onehot:
            feat_start,_ = self.featurizer_type_idxs["col_onehot"]
            # which column does the current feature belong to
            if col not in self.columns_onehot_idx:
                # print("{} not in columns_onehot_idx".format(col))
                return
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
            cmp_op, continuous, jobquery=False):

        ret_feats = []
        feat_idx_start = 0

        pfeats = np.zeros(self.max_pred_len)
        self._update_set_column_features(col, pfeats)

        # feat_idx_start += self.set_column_features_len

        # set comparison operator 1-hot value, same for all types
        cmp_start,_ = self.featurizer_type_idxs["cmp_op"]
        if cmp_op in self.cmp_ops_onehot:
            cmp_idx = self.cmp_ops_onehot[cmp_op]
            pfeats[cmp_start + cmp_idx] = 1.00

        col_info = self.column_stats[col]
        toaddpfeats = True
        if continuous:
            cstart,_ = self.featurizer_type_idxs["constant_continuous"]
            self._handle_continuous_feature(pfeats, cstart, col, val)
            ## will be done at the end
            # ret_feats.append(pfeats)
        else:
            if "like" in cmp_op:
                if not jobquery:
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
            if self.feat_separate_like_ests:
                assert pfeats[-3] == 0.0
                assert pfeats[-4] == 0.0
                if "like" in cmp_op:
                    pfeats[-4] = alias_est
                    pfeats[-3] = subp_est
                else:
                    pfeats[-2] = alias_est
                    pfeats[-1] = subp_est
            else:
                hstart,_ = self.featurizer_type_idxs["heuristic_ests"]
                if "synth" not in self.db_name:
                    pfeats[-2] = alias_est
                    assert pfeats[hstart] == alias_est

                pfeats[-1] = subp_est
                # test:
                assert pfeats[hstart+1] == subp_est

        if toaddpfeats:
            ret_feats.append(pfeats)

        return ret_feats

    def get_subplan_features_set(self, qrep, subplan, bitmaps=None,
            join_bitmaps=None,
            subset_edge=None):
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
        jobquery = "job" in qrep["template_name"]

        alltablefeats = []
        if self.table_features:
            ## table features
            # loop over each node, update the tfeats bitvector
            for alias in subplan:
                if alias == SOURCE_NODE:
                    continue

                tfeats = np.zeros(self.table_features_len)
                # need to find its real table name from the join_graph
                table = joingraph.nodes()[alias]["real_name"]
                table = table.replace(" ", "")

                if table not in self.seen_tabs:
                    # print("skipping unseen tabs {}".format(table))
                    alltablefeats.append(tfeats)
                    # pdb.set_trace()
                    continue

                if table not in self.table_featurizer:
                    print("table: {} not found in featurizer".format(table))
                    alltablefeats.append(tfeats)
                    pdb.set_trace()
                    # pdb.set_trace()
                    # assert False
                    continue
                # Note: same table might be set to 1.0 twice, in case of aliases
                tfeats[self.table_featurizer[table]] = 1.00

                # print(bitmaps)
                # pdb.set_trace()
                if self.sample_bitmap and bitmaps is not None \
                        and (alias,) in bitmaps:
                    # assert bitmaps is not None
                    startidx = len(self.table_featurizer)
                    # if alias not in bitmaps:
                        # continue
                    sb = bitmaps[(alias,)]

                    if self.sample_bitmap_key in sb:
                        bitmap = sb[self.sample_bitmap_key]
                        if self.feat_onlyseen_preds:
                            if table not in self.seen_bitmaps:
                                # print(table, " not in seen bitmaps")
                                # pdb.set_trace()
                                alltablefeats.append(tfeats)
                                continue
                            train_seenvals = self.seen_bitmaps[table]

                        # print("Bitmap for {} is: ".format(alias))
                        # print(bitmap)
                        # pdb.set_trace()
                        for val in bitmap:
                            if self.feat_onlyseen_preds:
                                if val not in train_seenvals:
                                    continue
                                bitmapidx = val % self.sample_bitmap_buckets
                                # bitmapidx = deterministic_hash(val) % self.sample_bitmap_buckets
                                tfeats[startidx+bitmapidx] = 1.0
                            else:
                                bitmapidx = val % self.sample_bitmap_buckets
                                # bitmapidx = deterministic_hash(val) % self.sample_bitmap_buckets
                                tfeats[startidx+bitmapidx] = 1.0
                    else:
                        pass

                alltablefeats.append(tfeats)

        # print(alltablefeats)
        # pdb.set_trace()
        featdict["table"] = alltablefeats

        alljoinfeats = []
        if self.join_features:
            ## this would imply the bitmap is the only feature
            if not self.join_bitmap:
                seenjoins = set()
                for alias1 in subplan:
                    for alias2 in subplan:
                        ekey = (alias1, alias2)
                        if ekey in joingraph.edges():
                            join_str = joingraph.edges()[ekey]["join_condition"]
                            join_str = self.join_str_to_real_join(join_str)

                            if join_str in seenjoins:
                                continue
                            if join_str not in self.seen_joins:
                                continue
                            seenjoins.add(join_str)
                            jfeats = self._handle_join_features(join_str)
                            alljoinfeats.append(jfeats)

            if self.join_bitmap:
                jfeats  = self._handle_join_bitmaps(subplan,
                        join_bitmaps, bitmaps, joingraph)
                alljoinfeats += jfeats

            if len(alljoinfeats) == 0:
                alljoinfeats.append(np.zeros(self.join_features_len))

        featdict["join"] = alljoinfeats

        allpredfeats = []

        for alias in subplan:
            if not self.pred_features:
                continue

            aliasinfo = joingraph.nodes()[alias]
            if "pred_cols" not in aliasinfo:
                continue

            if len(aliasinfo["pred_cols"]) == 0:
                continue

            node_key = tuple([alias])

            if self.true_base_cards:
                alias_est = self._get_true_est(subsetgraph.nodes()[node_key])
            else:
                alias_est = self._get_pg_est(subsetgraph.nodes()[node_key])

            subp_est = self._get_pg_est(subsetgraph.nodes()[subplan])

            if self.card_type == "joinkey":
                # TODO: find appropriate ones for this
                alias_est = alias_est
                assert alias_est <= 1.0
                subp_est = 0.0

            seencols = set()

            for ci, col in enumerate(aliasinfo["pred_cols"]):
                # we should have updated self.column_stats etc. to be appropriately
                # updated
                if not self.feat_separate_alias \
                        and "synth" not in self.db_name:
                    col = ''.join([ck for ck in col if not ck.isdigit()])

                if col not in self.column_stats:
                    continue

                allvals = aliasinfo["pred_vals"][ci]
                if isinstance(allvals, dict):
                    allvals = allvals["literal"]

                cmp_op = aliasinfo["pred_types"][ci]

                if cmp_op == "lt":
                    if len(aliasinfo["predicates"]) > ci:
                        if ">" in aliasinfo["predicates"][ci]:
                            cmp_op = ">"
                        elif "<" in aliasinfo["predicates"][ci]:
                            cmp_op = "<"

                # if jobquery and "like" in cmp_op.lower():
                    # # print("skipping featurizing likes for JOB")
                    # continue

                continuous = self.column_stats[col]["continuous"] \
                        and cmp_op in ["lt", "<", ">", "<=", ">="]

                if continuous and not isinstance(allvals, list):
                    # FIXME: hack for jobm queries like = '1997'
                    # print("Hacking around jobM: ", allvals)
                    allvals = [allvals, allvals]

                pfeats = self._handle_single_col(col,allvals,
                        alias_est, subp_est,
                        cmp_op,
                        continuous, jobquery=jobquery)

                # if subp_est != 0:
                    # print(pfeats)
                    # pdb.set_trace()

                # if "10M" in qrep["workload"]:
                    # print(col)
                    # print(pfeats)
                    # pdb.set_trace()

                if continuous:
                    pass
                    # print(aliasinfo)
                    # print(allvals)
                    # print(pfeats)

                allpredfeats += pfeats

        # print(allpredfeats)
        # pdb.set_trace()

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

        # assert len(allpredfeats) <= self.max_pred_vals
        if not len(allpredfeats) <= self.max_pred_vals:
            print(len(allpredfeats), self.max_pred_vals)
            pdb.set_trace()

        if self.pred_features:
            featdict["pred"] = allpredfeats
        else:
            featdict["pred"] = []

        global_features = []

        if self.global_features:
            global_features = self.get_global_features(subplan,
                    qrep["subset_graph"], qrep["template_name"],
                    qrep["join_graph"], subset_edge=subset_edge)

        featdict["flow"] = global_features

        return featdict

    def get_subplan_features_combined(self, qrep, subplan, bitmaps=None,
            join_bitmaps=None):
        assert isinstance(subplan, tuple)
        featvectors = []

        # we need the joingraph here because all the information about
        # predicate filters etc. on each of the individual tables is stored in
        # the joingraph; subsetgraph stores just the names of the
        # tables/aliases involved in a join
        subsetgraph = qrep["subset_graph"]
        joingraph = qrep["join_graph"]

        if self.table_features:
            ## table features
            # loop over each node, update the tfeats bitvector
            tfeats = np.zeros(self.table_features_len)

            for alias in subplan:
                if alias == SOURCE_NODE:
                    continue
                # tfeats = np.zeros(self.table_features_len)
                # need to find its real table name from the join_graph
                table = joingraph.nodes()[alias]["real_name"]
                table = table.replace(" ", "")

                if table not in self.seen_tabs:
                    continue

                if table not in self.table_featurizer:
                    print("table: {} not found in featurizer".format(table))
                    continue
                tidx = self.table_featurizer[table]
                tfeats[tidx] = 1.00

                if self.sample_bitmap and bitmaps is not None:
                    # assert bitmaps is not None
                    startidx = len(self.table_featurizer)
                    startidx += tidx*self.sample_bitmap_buckets

                    sb = bitmaps[(alias,)]
                    if self.sample_bitmap_key in sb:
                        bitmap = sb[self.sample_bitmap_key]
                        if self.feat_onlyseen_preds:
                            if table not in self.seen_bitmaps:
                                print(table, " not in seen bitmaps")
                                pdb.set_trace()
                                continue
                            train_seenvals = self.seen_bitmaps[table]

                        for val in bitmap:
                            if self.feat_onlyseen_preds:
                                if val not in train_seenvals:
                                    continue
                                bitmapidx = val % self.sample_bitmap_buckets
                                tfeats[startidx+bitmapidx] = 1.0
                            else:
                                bitmapidx = val % self.sample_bitmap_buckets
                                tfeats[startidx+bitmapidx] = 1.0
                    else:
                        pass

            featvectors.append(tfeats)

        if self.join_features:
            ## this would imply the bitmap is the only feature
            if self.join_bitmap:
                jfeats  = self._handle_join_bitmaps_combined(subplan,
                        join_bitmaps, bitmaps, joingraph)
                featvectors.append(jfeats)
            else:
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

                if cmp_op == "lt":
                    if len(aliasinfo["predicates"]) >= 1:
                        if ">" in aliasinfo["predicates"][0]:
                            cmp_op = ">"
                        elif "<" in aliasinfo["predicates"][0]:
                            cmp_op = "<"

                if col not in self.featurizer:
                    # print("col: {} not found in featurizer".format(col))
                    continue

                cmp_op_idx, num_vals, continuous = self.featurizer[col]
                if cmp_op in self.cmp_ops_onehot:
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
                if self.heuristic_features \
                        and len(subsetgraph.nodes()) > 1:
                    # assert pfeats[pred_idx_start + num_pred_vals-1] == 0.0
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

        if self.global_features:
            global_features = self.get_global_features(subplan,
                    qrep["subset_graph"], qrep["template_name"],
                    qrep["join_graph"])
            featvectors.append(global_features)

        feat = np.concatenate(featvectors)

        return feat

    def get_subplan_features_joinkey(self, qrep, subset_node, subset_edge,
            bitmaps=None,
            join_bitmaps=None):

        einfo = qrep["subset_graph"].edges()[subset_edge]
        if "join_key_cardinality" not in einfo:
            print("BAD!")
            print(subset_edge, einfo)
            pdb.set_trace()
            assert False

        joincols = list(einfo["join_key_cardinality"].keys())
        # assumption: all joins on same column
        joincol = joincols[0]

        assert self.featurization_type == "set"
        x = self.get_subplan_features_set(qrep,
                subset_node, bitmaps=bitmaps,
                join_bitmaps=join_bitmaps,
                subset_edge=subset_edge)

        # y-stuff
        true_val = einfo["join_key_cardinality"][joincol]["actual"]
        y = self.normalize_val(true_val, None)

        return x,y

    def get_subplan_features(self, qrep, node, bitmaps=None,
            join_bitmaps=None):
        '''
        @subsetg:
        @node: subplan in the subsetgraph;
        @ret: []
            will depend on if self.featurization_type == set or combined;
        '''
        # if self.sample_bitmap:
            # assert False, "TODO: not implemented yet"

        # the shapes will depend on combined v/s set feat types
        if self.featurization_type == "combined":
            x = self.get_subplan_features_combined(qrep,
                    node, bitmaps=bitmaps,
                    join_bitmaps = join_bitmaps)
        elif self.featurization_type == "set":
            x = self.get_subplan_features_set(qrep,
                    node, bitmaps=bitmaps,
                    join_bitmaps = join_bitmaps)
        else:
            assert False

        ## choosing the y values
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

    def get_global_features(self, node, subsetg,
            template_name, join_graph, subset_edge=None):

        assert node != SOURCE_NODE
        ckey = "cardinality"
        global_features = np.zeros(self.num_global_features, dtype=np.float32)
        cur_idx = 0

        if self.card_type == "joinkey":
            assert subset_edge is not None
            einfo = subsetg.edges()[subset_edge]
            joincols = list(einfo["join_key_cardinality"].keys())
            # assumption: all joins on same column
            joincol = joincols[0]
            pg_join_est = einfo["join_key_cardinality"][joincol]["expected"]

            pg_join_est = self.normalize_val(pg_join_est, None)

            joincol = "".join([jc for jc in joincol if not jc.isdigit()])

            realcol = self.join_col_map[joincol]
            jcol_idx = self.real_join_col_mapping[realcol]
            global_features[cur_idx + jcol_idx] = 1.0
            cur_idx += len(self.real_join_col_mapping)

        # incoming edges
        if self.global_feat_degrees:
            in_degree = subsetg.in_degree(node)
            in_degree = min(in_degree, self.max_in_degree)
            global_features[cur_idx + in_degree] = 1.0
            cur_idx += self.max_in_degree+1

            # outgoing edges
            out_degree = subsetg.out_degree(node)
            out_degree = min(out_degree, self.max_out_degree)
            global_features[cur_idx + out_degree] = 1.0
            cur_idx += self.max_out_degree+1

        if self.global_feat_tables:
            # # num tables
            max_table_idx = len(self.aliases)-1
            nt = len(node)
            # assert nt <= max_tables
            nt = min(nt, max_table_idx)
            global_features[cur_idx + nt] = 1.0
            cur_idx += len(self.aliases)

        # precomputed based stuff
        if self.feat_num_paths:
            if node in self.template_info[template_name]:
                num_paths = self.template_info[template_name][node]["num_paths"]
            else:
                num_paths = 0

            # assuming min num_paths = 0, min-max normalization
            global_features[cur_idx] = num_paths / self.max_paths
            cur_idx += 1

        if self.feat_pg_costs and self.heuristic_features and \
                self.cost_model is not None:
            in_edges = subsetg.in_edges(node)
            in_cost = 0.0
            for edge in in_edges:
                in_cost += subsetg[edge[0]][edge[1]][self.cost_model + "pg_cost"]
            # normalized pg cost
            global_features[cur_idx] = in_cost / subsetg.graph[self.cost_model + "total_cost"]
            cur_idx += 1

        if self.feat_tolerance:
            tol = subsetg.nodes()[node]["tolerance"]
            tol_idx = int(np.log10(tol))
            assert tol_idx <= 4
            global_features[cur_idx + tol_idx-1] = 1.0
            cur_idx += 4

        if self.feat_flows and self.heuristic_features:
            in_edges = subsetg.in_edges(node)
            in_flows = 0.0
            for edge in in_edges:
                in_flows += subsetg[edge[0]][edge[1]]["pg_flow"]
            # normalized pg flow
            global_features[cur_idx] = in_flows
            cur_idx += 1

        if self.feat_pg_path:
            if "pg_path" in subsetg.nodes()[node]:
                global_features[cur_idx] = 1.0

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
                global_features[cur_idx + tidx] = 1.0
            cur_idx += len(self.table_featurizer)

        if self.feat_rel_pg_ests and self.heuristic_features \
                and self.cost_model is not None:
            total_cost = subsetg.graph[self.cost_model+"total_cost"]
            pg_est = subsetg.nodes()[node][ckey]["expected"]
            global_features[cur_idx] = pg_est / total_cost
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
                global_features[cur_idx + tidx] = pg_est / ncard
                global_features[cur_idx + tidx] /= 1e5

            cur_idx += len(self.table_featurizer)

        if self.feat_rel_pg_ests_onehot \
                and self.heuristic_features \
                and self.cost_model is not None:
            total_cost = subsetg.graph[self.cost_model+"total_cost"]
            pg_est = subsetg.nodes()[node][ckey]["expected"]
            # global_features[cur_idx] = pg_est / total_cost
            pg_ratio = total_cost / float(pg_est)

            bucket = self.get_onehot_bucket(self.PG_EST_BUCKETS, 10, pg_ratio)
            global_features[cur_idx+bucket] = 1.0
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
                # global_features[cur_idx + tidx] = pg_est / ncard
                # global_features[cur_idx + tidx] /= 1e5
                if pg_est > ncard:
                    # first self.PG_EST_BUCKETS
                    bucket = self.get_onehot_bucket(self.PG_EST_BUCKETS, 10,
                            pg_est / float(ncard))
                    global_features[cur_idx+bucket] = 1.0
                else:
                    bucket = self.get_onehot_bucket(self.PG_EST_BUCKETS, 10,
                            float(ncard) / pg_est)
                    global_features[cur_idx+self.PG_EST_BUCKETS+bucket] = 1.0

                cur_idx += 2*self.PG_EST_BUCKETS

        if self.feat_pg_est_one_hot and self.heuristic_features:
            pg_est = subsetg.nodes()[node][ckey]["expected"]

            for i in range(self.PG_EST_BUCKETS):
                if pg_est > 10**i and pg_est < 10**(i+1):
                    global_features[cur_idx+i] = 1.0
                    break

            if pg_est > 10**self.PG_EST_BUCKETS:
                global_features[cur_idx+self.PG_EST_BUCKETS] = 1.0
            cur_idx += self.PG_EST_BUCKETS

        if self.card_type == "subplan":
            pg_est = self._get_pg_est(subsetg.nodes()[node])
            # if "expected" not in subsetg.nodes()[node]["cardinality"]:
                # print(node)
                # pdb.set_trace()
            # pg_est = subsetg.nodes()[node]["cardinality"]["expected"]
            # try:
                # total = subsetg.nodes()[node]["cardinality"]["total"]
            # except:
                # total = None
            # pg_est = self.normalize_val(pg_est, total)

        elif self.card_type == "joinkey":
            pg_est = pg_join_est
        else:
            assert False

        # assert global_features[-1] == 0.0
        if global_features[-1] != 0.0:
            print("BAD last val")
            print(global_features)
            pdb.set_trace()

        if self.heuristic_features:
            global_features[-1] = pg_est

        return global_features

    def unnormalize_torch(self, y, total):
        if self.ynormalization == "log":
            est_cards = torch.exp((y + self.min_val)*(self.max_val-self.min_val))
            ## wrong one??
            # est_card = np.exp((y*(self.max_val-self.min_val) + self.min_val))
        elif self.ynormalization == "selectivity":
            est_cards = y*total
        elif self.ynormalization == "selectivity-log":
            est_cards = (torch.exp(y)) * total
        else:
            assert False
        return est_cards

    def unnormalize(self, y, total):
        if self.ynormalization == "logwhitening":
            est_card = np.exp((y*self.stdy) + self.meany)
        elif self.ynormalization == "whitening":
            est_card = (y*self.stdy) + self.meany
            if est_card <= 0:
                est_card = 1
        elif self.ynormalization == "log":
            ## wrong?
            est_card = np.exp((y*(self.max_val-self.min_val) + self.min_val))

            # est_cards = torch.exp((y + \
                # self.min_val)*(self.max_val-self.min_val))
            # est_card = np.exp((y + self.min_val)*(self.max_val-self.min_val))

        elif self.ynormalization == "minmax":
            est_card = (float(y) * (self.max_val-self.min_val)) + self.min_val
        elif self.ynormalization == "selectivity":
            y = (y + self.min_val)*(self.max_val-self.min_val)
            est_card = y*total

        elif self.ynormalization == "selectivity-log":
            est_card = (np.exp(y)) * total

        elif self.ynormalization == "log-selectivity":
            y = y*np.log(total)
            est_card = np.exp(y)
        else:
            assert False

        # if est_card == 0:
            # est_card += 1
        # print(est_card)

        return est_card

    def normalize_val(self, val, total):
        if val == 0:
            val += 1
        if val < 0:
            val = 1

        if self.ynormalization == "logwhitening":
            return (np.log(float(val)) - self.meany) / self.stdy
        elif self.ynormalization == "whitening":
            return (float(val) - self.meany) / self.stdy
        elif self.ynormalization == "log":
            return (np.log(float(val)) - self.min_val) / (self.max_val-self.min_val)
        elif self.ynormalization == "minmax":
            ret =  (float(val) - self.min_val) / (self.max_val-self.min_val)
            return ret
        # elif self.ynormalization == "selectivity":
            # return float(val) / total

        elif self.ynormalization == "selectivity":
            sel = float(val) / total
            return (sel - self.min_val) / (self.max_val-self.min_val)

        elif self.ynormalization == "selectivity-log":
            sel = float(val) / total
            if sel == 0:
                sel += 1
            logsel = np.log(sel)
            return logsel
        elif self.ynormalization == "log-selectivity":
            logsel = np.log(float(val)) / np.log(total)
            # if logsel == 0:
                # logsel += 1
            return logsel
        else:
            assert False

    def _update_mcvs(self, column):
        # TODO: just need table+column w/o alias here
        if column in self.mcvs:
            return

        table = column[0:column.find(".")]
        attr_name = column[column.find(".")+1:]
        table = table.replace(" ", "")

        if table in self.aliases:
            table_real_name = self.aliases[table]
        else:
            return

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

        # total_count_query = COUNT_SIZE_TEMPLATE.format(FROM_CLAUSE =
                # '"' + table_real_name + '"')
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
            if "pred_cols" not in info:
                continue

            for col in info["pred_cols"]:
                col = col.replace(" ", "")
                cur_columns.append(col)

            if "implied_pred_cols" in info:
                for col in info["implied_pred_cols"]:
                    cur_columns.append(col)

        joins = extract_join_clause(qrep["sql"])

        for join in joins:
            join = join.replace(" ", "")
            keys = join.split("=")
            keys.sort()

            # join = self.join_str_to_real_join(join)
            # print(join)

            # keystr = ",".join(keys)
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

                # curtab = '"' + curtab + '"'

                # print("skipping join stats for: ", jkey)
                # continue
                print(curcol)
                csplit_idx = curcol.find(".")
                curcol = curcol[0:csplit_idx+1] + '"' + curcol[csplit_idx+1:]+'"'

                for si,tmp in enumerate(self.join_key_stat_tmps):
                    sname = self.join_key_stat_names[si]
                    execcmd = tmp.format(TABLE=curtab,
                                         COL=curcol)
                    try:
                        val = float(self.execute(execcmd)[0][0])
                    except:
                        val = 0.0
                    self.join_key_stats[jkey][sname] = val

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
            column = column.replace(" ", "")
            if column in self.column_stats:
                continue

            # FIXME: reusing join key code here
            jkey = column
            self.join_key_stats[jkey] = {}
            curalias = jkey[0:jkey.find(".")]
            curalias = curalias.replace(" ", "")
            curcol = jkey[jkey.find(".")+1:]
            curcol = curcol.replace(" ", "")
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

            ## not using this for now
            # self._update_mcvs(column)

            table = column[0:column.find(".")]
            table = table.replace(" ", "")

            column_stats = {}
            if table in self.aliases:
                table = ALIAS_FORMAT.format(TABLE = self.aliases[table],
                                    ALIAS = table)
            else:
                print(table)
                pdb.set_trace()

            table = '"' + table + '"'

            csplit_idx = column.find(".")
            column = column[0:csplit_idx+1] + '"' + column[csplit_idx+1:]+'"'

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
