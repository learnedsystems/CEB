# Cardinality Estimation Benchmark

## Contents
  * [Setup](#setup)
      - [Workload](#workload)
      - [PostgreSQL](#postgresql)
        - [Docker](#docker)
        - [Virtualbox](#virtualbox)
        - [Local Setup](#local-setup)
      - [Python requirements](#python-requirements)
  * [Usage](#usage)
      - [Query Representation](#query-representation)
      - [Evaluating Estimates](#evaluating-estimates)
      - [Getting Runtimes](#getting-runtimes)
      - [Visualizing Results](#visualizing-results)
      - [Learned Models](#learned-models)
        - [Examples](#examples)
        - [Featurization](#featurization)
        - [Featurization Knobs](#featurization-knobs)
        - [Loss Functions](#loss-functions)

      - [Generating Queries](#generating-queries)
      - [Generating Cardinalities](#generating-cardinalities)
  * [Future Work](#futurework)
  * [License](#license)

## Setup

If you are only interested in evaluating the cardinalities, using a loss
function such as Q-Error, or if you just want to use the queries for some other task, then you just need to download the workload. But the main goal of this dataset is to make it easy to evaluate the impact of cardinality estimates on query optimization. For this, we use PostgreSQL (and eventually plan to add support for other open source DBMS' like MySQL). We provide a dockerized setup with the appropriate setup to get started right away; Instead, you can also easily adapt it to your own installation of PostgreSQL. Docker is the easiest way to started with CEB.

### PostgreSQL

#### Docker

We use docker to install and configure PostgreSQL, and setup the relevant databases. Make sure that you have Docker installed appropriately for your system, with the docker daemon running. PostgreSQL requires a username, which we copy from an environement variable $LCARD_USER while setting it up in docker. Similarly, set $LCARD_PORT to the local port you want to use to connect to the PostgreSQL instance running in docker. Here are the commands to set it up:

```bash
cd docker
export LCARD_USER=ceb
export LCARD_PORT=5432
sudo docker build --build-arg LCARD_USER=${LCARD_USER} -t pg12 .
sudo docker run -itd --shm-size=1g --name card-db -p ${LCARD_PORT}:5432 -d pg12
sudo docker restart card-db
sudo docker exec -it card-db /imdb_setup.sh
```

Note: Depending on the settings of your docker instance, you may not require sudo in the above commands. Also, in the docker run command, you may want to choose the --shm-size parameter depending on your requirements.

<b> Optionally </b> you can use the following command to install the stackexchange database; But the stackexchange database is A LOT larger than the IMDb database --- make sure you have up to 150GB space on your device before running the following command.

```bash
sudo docker exec -it card-db /stack_setup.sh

# These commands ensure there are only foreign key : primary key
# indexes on the stackexchange database; Without the drop_indexes.sql command,
#the database contains a few indexes that utilize multiple columns, which may
#potentially be better suited for the join topology in the stackexchange
#database, but which index setup is the most appropriate remains to be explored further.

psql -d stack -U $LCARD_USER -p $LCARD_PORT -h localhost < drop_indexes.sql
psql -d stack -U $LCARD_USER -p $LCARD_PORT -h localhost < create_indexes.sql
```

The StackExchange database was based on one of the dumps released from the
StackExchange foundation; We've used various heuristics / simplifications in
constructing the database from the StackExchange dump, and restore the database
from a PostgreSQL snapshot (see stack_setup.sh for its download link).
The StackExchange database holds a lot of potential to develop more challenging query templates as well, although we have not explored it as much as IMDb. Refer to the [workload](#workload) section for a comparison between IMDb and StackExchange workloads.

Here are a few useful commands to check / debug your setup:
```bash
# if your instance gets restarted / docker image gets shutdown
sudo docker restart card-db

# get a bash shell within the docker image
sudo docker exec -it card-db bash
# note that postgresql data on docker is stored at /var/lib/postgresql/data

# connect psql on your host to the postgresql server running on docker
psql -d imdb -h localhost -U imdb -p $LCARD_PORT
```

To clean up everything, if you don't want to use the docker image anymore, run:
```bash
bash clean.sh
```

#### Virtualbox
Follow instructions provided [here](https://github.com/RyanMarcus/imdb_pg_dataset).
After setting up the database, you should be able to use the scripts here by
passing in the appropriate user, db_name, db_host, and ports to appropriate python
function calls.

### Workload

Our goal is to eventually add multiple database / query workloads to evaluate
these models; CEB contains IMDb and StackExchange databases. Moreover, we
provide other known workloads on IMDb, like the Join Order Benchmark (JOB), or
its simpler versions JOB-M, JOB-light in the same format as well, which makes
it easy to use the various tools for computing plan costs, runtimes etc. on
those queries (see next sections for the description of the CEB format etc).

Note that you only need to download one of these workloads in order to get
started with CEB, and can choose which one fits your needs best.

#### IMDb CEB

Download the full IMDb CEB workload to `queries/imdb`.

```bash
bash scripts/download_imdb_workload.sh
```

Each directory represents a query template. Each query, and all it's subplans, is represented using a pickle file, of the form `1a100.pkl`. This workload has over 13k queries; for most purposes, especially when testing out new models, you should probably use a smaller subset of the workload as evaluating on the whole dataset can take more time. For instance, we provide flags to run on only some templates, or to have only up to N queries per template.

One useful subset of the data is by considering the PostgreSQL query plans when
using true cardinalities; We can deduplicate all the queries where the true
cardinalities map to the same query plan. This has about 3k queries; We have not explored the difference in the model performance' in these two scenarios, but for most practical purposes, this should be a sufficiently large dataset as well. We can download this by:

```bash
bash scripts/download_imdb_uniqueplans.sh
```

#### JOB

```bash
bash scripts/download_job_workloads.sh
```

This will download both the JOB and JOB-M workloads to the queries/job or
queries/jobm directories. In terms of the various Plan-Cost metrics (see
    Section [Evaluating Estimates](#evaluating-estimates)), these workloads are
somewhat less challenging than CEB, but do have a few non-trivial queries where
cardinality estimates become very important.

For the JOB-light workload, see the [generating
cardinalities](#generating-cardinalities) section. Since the JOB-light workload
has relatively small queries, it serves as a nice example of the tools to take
in input sqls and generate the cardinalities of all the subplans, and store
them in the format we support. As a drawback, even PostgreSQL estimates do very
well on JOB-light in terms of the Plan Cost metrics, thus it is not very
challenging from the perspective of query optimization.

#### StackExchange CEB

```bash
bash scripts/download_stack_workload.sh
```

### Python Requirements

These can be installed with

```bash
pip3 install -r requirements.txt
```

To test the whole setup, including the docker installation, run

```bash
python3 tests/test_installation.py
```

## Usage

### Query Representation

First, let us explore the basic properties of the queries that we store:

```python
from query_representation.query import *

qfn = "queries/imdb/4a/4a100.pkl"
qrep = load_qrep(qfn)

# extract basic properties of the query representation format

print("""Query has {} tables, {} joins, {} subplans.""".format(
    len(qrep["join_graph"].nodes()), len(qrep["join_graph"].edges()),
    len(qrep["subset_graph"].nodes())))

tables, aliases = get_tables(qrep)

print("Tables: ")
for i,table in enumerate(tables):
    print(table, aliases[i])

print("Joins: ")
joins = get_joins(qrep)
print(("\n").join(joins))

preds, pred_cols, pred_types, pred_vals = get_predicates(qrep)
print("Predicates: ")
for i in range(len(preds)):
    for j, pred in enumerate(preds[i]):
        print(pred.strip(" "))
        print("     Predicate column: ", pred_cols[i][j])
        print("       Predicate type: ", pred_types[i][j])
        print("     Predicate values: ", pred_vals[i][j])
```

Next, we see how to access each of the subplans, and their cardinality
estimates.

```python
from query_representation.query import *

qfn = "queries/imdb/4a/4a100.pkl"
qrep = load_qrep(qfn)

# for getting cardinality estimates of every subplan in the query
ests = get_postgres_cardinalities(qrep)
trues = get_true_cardinalities(qrep)

for k,v in ests.items():
    print("Subplan, joining tables: ", k)
    subsql = subplan_to_sql(qrep, k)
    print("Subplan SQL: ", subsql)
    print("   True cardinality: ", trues[k])
    print("PostgreSQL estimate: ", v)
    print("****************")
```

Please look at the implementations in query_representation/queries.py for seeing how the information is represented, and how to directly manipulate the internal fields of the qrep object.

A few other points to note:
  * all queries uses table aliases in the workload (e.g., TITLE as t). A lot of the helper methods for generating cardinalities etc. assume this, so if you want to use these tools to generate data for new queries, use aliases.


### Evaluating estimates

Given a query, and estimates for each of its subplans, we can use various error
functions to evaluate how good the estimates are. We can directly compare the
true values and the estimated values, using for instance:
  * Q-Error, Relative Error, Absolute Error etc. Q-Error is generally considered
  to be the most useful of these metrics from the perspective of
  query-optimization.

Alternatively, we can compare how good was the plan generated by using the
estimated values. This will depend on the query optimizer - in particular the
properties of the cost model we choose, and the search function etc. We provide
implementations for the two options as discussed in the paper, but by changing
configurations of the PostgreSQL cost model, or adding more complex custom
cost models, there can be many possibilities considered here.

  * Postgres Plan Cost (PPC): this uses the PostgreSQL cost model with two
                              restrictions --- no materialization and
                              parallelism. For experimenting with different
                              configurations, check the function set_cost_model
                              in losses/plan_loss.py and add additional
                              configurations.

  * Plan-Cost: this considers only left deep plans, and uses a simple user
               specified cost function (referred to as C in the paper).

Here is a self contained example showing the API to compute these different
kind of errors on a single query.

```python
from query_representation.query import *
from evaluation.eval_fns import *

qfn = "queries/imdb/4a/4a100.pkl"
qrep = load_qrep(qfn)
ests = get_postgres_cardinalities(qrep)

# estimation errors for each subplan in the query
qerr_fn = get_eval_fn("qerr")
abs_fn = get_eval_fn("abs")
rel_fn = get_eval_fn("rel")

qerr = qerr_fn.eval([qrep], [ests])
abs_err = abs_fn.eval([qrep], [ests])
relerr = rel_fn.eval([qrep], [ests])

print("avg q-error: {}, avg abs-error: {}, avg relative error: {}".format(
              np.round(np.mean(qerr),2), np.round(np.mean(abs_err), 2),
                            np.round(np.mean(relerr), 2)))

# check the function comments to see the description of the arguments
# can change the db arguments appropriately depending on the PostgreSQL
# installation.
ppc = get_eval_fn("ppc")
ppc = ppc.eval([qrep], [ests], user="ceb", pwd="password", db_name="imdb",
        db_host="localhost", port=5432, num_processes=-1, result_dir=None,
        cost_model="cm1")

# we considered only one query, so the returned lists have just one element
print("PPC is: {}".format(np.round(ppc[0])))

pc = get_eval_fn("plancost")
plan_cost = pc.eval([qrep], [ests], cost_model="C")
print("Plan-Cost is: {}".format(np.round(plan_cost[0])))
```

For evaluating either true cardinalities, or PostgreSQL estimates on all queries in CEB / or just from some templates, run:

```bash
python3 main.py --query_templates 1a,2a --algs true,postgres --eval_fns qerr,ppc,plancost --query_dir queries/imdb
```

Similarly, if you have setup JOB, JOB-M in the previous steps, you can run:

```bash
python3 main.py --query_templates 1a,2a --algs true,postgres --eval_fns qerr,ppc,plancost --query_dir queries/job
python3 main.py --query_templates 1a,2a --algs true,postgres --eval_fns qerr,ppc,plancost --query_dir queries/jobm
```
Since, JOB (and JOB-M), have only 2-4 queries per template, we do not separate
them out by templates.

If you have setup the StackExchange DB and workload, then you can run a similar
command, but passing the additional required parameters for the db\_name:

```bash
python3 main.py --query_templates all --algs true,postgres --eval_fns qerr,ppc,plancost --query_dir queries/stack --db_name stack
```

#### Notes on Postgres Plan Cost

* What is a good cost? This is very context dependent; What we really care
about is runtimes, but plan costs are nice proxies for them because they can be
computed quickly, and repeatedly. But these costs don't have units. But, it can
be helpful to compare them with the Postgres Plan Cost generated using the true
cardinality as estimators (use flag: --algs true).

* We can normalize these Postgres Plan Costs by dividing by the optimal cost OR
subtracting the optimal cost; But it is not clear what is the best approach;
For instance, it is common to see that some templates in CEB (e.g., 3a,4a),
    have much lower magnitude of PPC (e.g., < 100k), while some templates have
    PPC in the order of > 1e7; For a particular query, if the optimal cost is
    20k, and your estimator's cost is 40k, then this would lead to a relative
    error of 2.0; While for a large query, with an optimal cost of 1e7, and the
    estimator's cost of 1.5e7, would lead to a relative error of 1.5. But in
    these cases, as expected, we find the runtime differences to be more
    prominent in the latter case.

#### Adding other DBMS backends
We can similarly define MySQL Plan Cost, and plan costs using other database
backends. Plan costs are computed using the following steps:

1. Insert cardinality estimates into the optimizer; Get the output plan. (Note
     that since we used the estimated cardinalities, the cost we get for this plan does not mean much).
1. Insert the true cardinality estimates into the optimizer; Cost the plan from
   the previous step using the true cardinalities. This reflects how `good`
   that plan is in reality; we call it the DBMS Plan Cost.

These require two abilities: inserting cardinality estimates into an optimizer
(typically, not provided by most optimizers), and then precisely specifiying a
plan (for the step 2 above); It is usually possible to precisely specify a
plan, although this can be tricky. An example of this process with PostgreSQL
is in the file evaluation/plan\_losses.py; look at the function
\_get_pg_plancosts;

We've a mysql [fork](https://github.com/parimarjan/mysql-server) that
implements these requirements; But it is somewhat hacky and not easy to use. We
plan to eventually add a cleaner version to this repo. Similarly, you can add
other database backends as well.

### Getting runtimes

There are two steps to generating the runtimes; first, we generate the Postgres
Plan Cost, and the corresponding SQLs to execute. These SQL strings would be
annotated with various pg_hint_plan hints to enforce join order, operator
selection and index Postgres Plan Costselection (see losses/plan_losses.py for
    details). These strings can be executed on PostgreSQL with pg_hint_plan
loaded, but you may want to use a different setup for execution --- so other
processes on the computer do not interfere with the execution times, and do
things like clear the cache after every execution (cold start), or repeat each
execution a few times etc.  depending on your goals. Here, we provide a simple
example to execute the SQLs, but note that this  does not clear caches, or take
care about isolating the execution from other processes, so these timings won't
be reliable.

```bash
# writes out the file results/Postgres/PostgresPlanCost.csv with the sqls to execute
# the sqls are modified with pg_hint_plan hints to use the cardinalities output
# by the given algorithm' estimates;
python3 main.py --algs postgres -n 5 --query_template 1a --eval_fns qerr,ppc,plancost

# executes the sqls on PostgreSQL server, with the given credientials
python3 evaluation/get_runtimes.py --port 5432 --user ceb --pwd password --result_dir results/Postgres
```

In the [Flow-Loss paper](http://vldb.org/pvldb/vol14/p2019-negi.pdf), we
executed these plans on AWS machines w/ NVME hard disks, using the code in this [repo](http://github.com/parimarjan/prism-testbed/). It is not clear what is the best environment to evaluate runtime of these plans, and you should choose the appropriate settings for your project.

### Visualizing Results

When you execute,
```bash
python3 main.py --algs postgres -n 5 --query_template 1a --eval_fns qerr,ppc
```

it should create files results/Postgres/train_query_plans.pdf etc. which
contain a pdf with the query plan generated using the cardinalities from
PostgreSQL, and additional details.

For instance, consider the query 11c in the Join Order Benchmark. The query
plan based on PostgreSQL estimates is 4-5x worse than the query plan based on
the true cardinalities; We can clearly see why the PostgreSQL estimates mess up
by looking at the accompanying visualization ![plot](images/job-11c-postgres.png?raw=true "Join Order Benchmark, 11c")

Note that the cardinalities are rounded to the nearest thousand. The red node is the most expensive cost node; PostgreSQL was estimating a very low cardinality (blue), and went for a nested loop join. The true cardinality (orange), is larger, and a Hash Join was probably a better choice (if you check the same plan with flag --algs true , you will see that the best plan does use a similar plan, but with Hash Join in the third join).

### Learned Models

#### Examples

We provide baseline implementation of a few common learning algorithms. Here
are a few sample commands to run these:

```bash
python3 main.py --query_templates 1a,2a --algs xgb --eval_fns qerr,ppc,plancost --result_dir results
python3 main.py --query_templates 1a,2a --algs mscn --eval_fns qerr,ppc,plancost --result_dir results --lr 0.0001
python3 main.py --query_templates all --algs fcnn --eval_fns qerr,ppc,plancost --result_dir results --lr 0.0001
```

Please look at cardinality\_estimation/algs.py for the list of provided
implementations.

#### Train Test Split

We suggest two ways to split the dataset; `--train_test_split_kind query`
splits train/test samples among the queries on each template. So for instance,
if we run the following

```bash
python3 main.py --query_templates all --algs fcnn --eval_fns qerr,ppc,plancost --result_dir results --train_test_split_kind query --val_size 0.2 --test_size 0.5
```

the, for each template, we will select 0.2% queries in the validation set, and
divide the remaining equally into train and test sets.

A more challenging scenario will be to have a few unseen templates. For this,
  use the flag `--train_test_split_kind template`. For example:

```bash
python3 main.py --query_templates all --algs fcnn --eval_fns qerr,ppc,plancost --result_dir results --train_test_split_kind template --test_size 0.5 --diff_templates_seed 1
```

This will divide the templates equally into train / test sets; Note: we can
have a validation set in this case as well, but since the templates
performance' can be very different from each other, its not clear if the
validation set helps much. Since there are relatively few templates, the exact
split can create very different experiments, thus, in this scenario we suggest
cross validating across multiple such splits (e.g., by using
    --diff_templates_seed 1, --diff_templates_seed 2, etc.).

#### Flattened 1d v/s Set features

The featurization scheme is implemented in cardinality\_estimation/featurizer.py. Look at the keyword arguments for Featurizer.setup() for the various configurations we use to generate the 1-d featurization of the queries.

The goal is to featurize each subplan in a query, so we can make cardinality
predictions on them; There are two featurization approaches.

<b> Flattened </b> This flattens a given subplan into a single 1d feature vector. This can be very convenient as most learning methods operate on 1d arrays; As a drawback, it requires reserving a spot in this feature vector for each column used in the workloads; Thus, you need to know the exact workloads the query will be evaluated on (this is not uncommon, as templated / dashboard queries would match such workloads, but it does reduce the scope of such classifiers.

<b> Set </b> This maps a given subplan into three sets (table, predicate,
  joins) of feature vectors; For instance, the tables set will contain one
feature vector for each of the tables in the subplan, and the predicate set
will contain one feature vector for each of the predicate filters in the
subplan. If a subplan has three tables, five filter predicates, and two joins,
it will map the subplan into a (3,TABLE_FEATURE_SIZE), (5, PREDICATE_FEATURE_SIZE), (2, JOIN_FEATURE_SIZE) vectors. Notice, for e.g., with predicates, the featurizer only needs to map a filter predicate on a single column to a feature vector; This is used in the set based MSCN architecture, which learns on pooled aggregations of different sized sets like these. The main benefit is that it avoids potentially massively large flattened 1d feature vectors, and can support a larger class of ad-hoc queries. The drawback is that different subplans have different number of tables, predicates etc. And in order to get the MSCN implemenations working, we need to pad these with zeros so all `sets' are of same size, thus it requires A LOT of additional memory spent on just padding small subplans, e.g., a subplan which has 1 predicate filter will have size (1,PREDICATE_FEATURE_SIZE). But to pass it to the MSCN architecture, we'll need to pad it with zeros so its size will be (MAX_PREDICATE_FILTERS, PREDICATE_FEATURE_SIZE), where the MAX_PREDICATE_FILTERS is the largest value it is in the batch we pass into the MSCN model. We have not explored optimizing / using sparse representations for the MSCN architecture, which can perhaps save these memory cost blowups.

Using the flag --load_padded_mscn_feats 0 will only load the non-zero sets, and
pad them as needed. The current implementation is very slow, and it is almost
10x slower to train like this (we can use parallel dataloaders, or implement
  this step in C etc. to speed this up considerably). --load_padded_mscn_feats 1 will pre-load all these padded sets into memory (very unpractical on the full CEB workload), but runs fast.

#### Featurization Knobs

There are several assumptions that go into the featurization step, and a lot of
scope for improvement. I'll describe the most interesting ones here:

* <b> PostgreSQL estimates </b> We find that these estimates significantly
improve model performance (although they do slow inference times in a practical
    system, as the featurization will require calling the PostgreSQL estimator;
    the impact of this slowdown needs to be studied)

* <b> Featurizing Categorical Columns </b>: It's easy to featurize continuous
columns; What about =, or IN predicates? Its unclear what is the best approach
in general. Providing heuristic estimates, like PostgreSQL estimates, is a
general approach, but might not be as useful. Sample Bitmaps is another
approach that can be quite informative for the model, but these are large 1d
vectors, which significantly increase the size of feature vectors / memory
requirements etc.; We take the approach of hashing the predicate filter values
to a small array, whose size is controlled by the feature: --max_discrete_featurizing_buckets; Intuitively, if the number of unique values in the column is not too much larger than the number of buckets we use, then this can be a useful signal for workloads with repeating patterns; In CEB, we typically had (IN, =) predicates on columns with not very large alphabet sizes, so --max_discrete_featurizing_buckets 10 significantly improves the performance of these models;

But its debatable if this is scalable to more challenging workloads. Thus, in general, it may be interesting to develop methods, and evaluate models, while setting --max_discrete_featurizing_buckets 1; In a previous paper, [Cost Guided Cardinality Estimation](https://ieeexplore.ieee.org/document/9094107), we used this setting. Methods that can improve model performance when max_discrete_featurizing_buckets == 1, might perhaps model a more common scenario where the alphabet size of categorical columns is too large, so these features become less useful.

#### Loss Functions

The standard approach is to optimize the neural networks for Q-Error; One way
to implement this is to first take the log of the target values by (use flag:
  --ynormalization log), and then optimize for mean squared error. As shown in
the [paper](http://www.vldb.org/pvldb/vol12/p1044-dutt.pdf), this is close
enough to optimizing for Q-Error, and has some nicer properties (smoother
    function than Q-Error, which may help in training). In practice, we find
this to work slightly better in general than optimizing for q-error directly.

### Generating Queries

Queries in CEB are generated based on templates. Example templates are in the
directory /templates/. More details about the templating scheme, including the
base SQL structure of the templates in the IMDb workload are given here
[templates](/TEMPLATES.md). For generating queries from such a template, you
can use:

```bash
python3 query_gen/gen_queries.py --query_output_dir qreps --template_dir ./templates/imdb/3a/ -n 10 --user ceb --pwd password --db_name imdb --port 5432
```

### Generating Cardinalities (TODO)

Here, we will provide an example that shows how to go from a bunch of sql files
to the qrep objects which contain all the cardinality estimates for subplans,
   and is the format used to represent queries in this project. As a simple
   example, we have added the JOB-light queries in the repo; these have small
   join graphs, thus, the cardinalities can be generated very fast.

```bash
# Change the input / output directories appropriately in the script etc.
python3 scripts/sql_to_qrep.py

# this updates all the subplans of each qrep object with the postgresql
# estimates that we use in featurization etc. of the subplans. This is stored in
# the field \[subplan\]\["cardinality"\]\["expected"\]
python3 scripts/get_query_cardinalities.py --port 5432 --db_name imdb --query_dir queries/joblight/all_joblight/ --card_type pg --key_name expected --pwd password --user ceb

# this updates all the subplans of each qrep object with the actual
# estimates that we use in featurization etc. of the subplans. This is stored in
# the field \[subplan\]\["cardinality"\]\["actual"\]
# This step could take really long depending on the size of the query, the
# evaluation setup you have etc. Also, be careful with the resource utilization
#by this script: by default, it parallelizes the executions, but this might
#cause PostgreSQL to crash in case there is not enough resources (check flags
#    --no_parallel 1 to do it one query at a time)
python3 scripts/get_query_cardinalities.py --port 5432 --db_name imdb --query_dir queries/joblight/all_joblight/ --card_type actual --key_name actual --pwd password --user ceb
```

### TODO: Using wanderjoin

## Future Work
