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
      - [Learned Models](#learned-models)
      - [Generating Queries](#generating-queries)
      - [Generating Cardinalities](#generating-cardinalities)
  * [Future Work](#futurework)
  * [License](#license)

## Setup

If you are only interested in evaluating the cardinalities, using a loss
function such as Q-Error, or if you just want to use the queries for some other task, then you just need to download the workload. But the main goal of this dataset is to make it easy to evaluate the impact of cardinality estimates on query optimization. For this, we use PostgreSQL. We provide a dockerized setup with the appropriate setup to get started right away; Instead, you can also easily adapt it to your own installation of PostgreSQL.

Docker is the easiest way to started with CEB. But, for a long term project, it is probably better to use a system wide installation of PostgreSQL.

### Workload

Download the IMDb CEB workload to `queries/imdb`.

```bash
bash scripts/download_imdb_workload.sh
```

Each directory represents a query template. Each query, and all it's sub-plans, is represented using a pickle file, of the form `1a100.pkl`.

### PostgreSQL

#### Docker

We use docker to install and configure PostgreSQL, and setup the relevant databases. Make sure that you have Docker installed appropriately for your system, with the docker daemon running. PostgreSQL requires a username, which we copy from an environement variable $LCARD_USER while setting it up in docker. Similarly, set $LCARD_PORT to the local port you want to use to connect to the PostgreSQL instance running in docker. Here are the commands to set it up:

```bash
cd docker
export LCARD_USER=imdb
export LCARD_PORT=5432
docker build --build-arg LCARD_USER=${LCARD_USER} -t pg12 .
docker run --name card-db -p ${LCARD_PORT}:5432 -d pg12
docker restart card-db
docker exec -it card-db /imdb_setup.sh
```

Note: Depending on the settings of your docker instance, you may require sudo
in the above commands.

Here are a few useful commands to check / debug your setup:
```bash
# if your instance gets restarted / docker image gets shutdown
docker restart card-db

# get a bash shell within the docker image
docker exec -it card-db bash
# note that postgresql data on docker is stored at /var/lib/postgresql/data

# connect psql on your host to the postgresql server running on docker
psql -d imdb -h localhost -U imdb -p $LCARD_PORT
```

To test this, run
```bash
python3 tests/test_installation.py
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

#### Local Setup (TODO)

### Python Requirements

These can be installed with

```bash
pip3 install -r requirements.txt
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

Next, we see how to access each of the sub-plans, and their cardinality
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
ppc = ppc.eval([qrep], [ests], user="imdb", pwd="password", db_name="imdb",
        db_host="localhost", port=5432, num_processes=-1, result_dir=None,
        cost_model="cm1")

# we considered only one query, so the returned lists have just one element
print("PPC is: {}".format(np.round(ppc[0])))

pc = get_eval_fn("plancost")
plan_cost = pc.eval([qrep], [ests], cost_model="C")
print("Plan-Cost is: {}".format(np.round(plan_cost[0])))
```

For evaluating postgres on all queries in CEB / or just from some templates,
run:

```bash
python3 main.py --query_templates 1a,2a --algs postgres --eval_fns qerr,ppc --result_dir results
```
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
python3 main.py --algs postgres -n 5 --query_template 1a --eval_fns qerr,ppc,plancost --result_dir results

# executes the sqls on PostgreSQL server, with the given credientials
python3 evaluation/get_runtimes.py --port 5432 --user imdb --pwd password --result_dir results/Postgres
```

TODO: describe the execution setup used in the paper.

### Learned Models

We provide baseline implementation of a few common learning algorithms. Here
are a few sample commands to run these:

```bash
python3 main.py --query_templates 1a,2a --algs xgb --eval_fns qerr,ppc,plancost --result_dir results
python3 main.py --query_templates 1a,2a --algs mscn --eval_fns qerr,ppc,plancost --result_dir results --lr 0.0001
python3 main.py --query_templates all --algs fcnn --eval_fns qerr,ppc,plancost --result_dir results --lr 0.0001
```

Please look at cardinality\_estimation/algs.py for the list of provided
implementations.

The featurization scheme is implemented in
cardinality\_estimation/featurizer.py. Look at the keyword arguments for
Featurizer.setup() for the various configurations we use to generate the 1-d
featurization of the queries.

### Generating Queries

Queries in CEB are generated based on templates. Example templates are in the
directory /templates/. More details abotu the templating scheme, including the
base SQL structure of the templates in the IMDb workload are given here
[templates](/TEMPLATES.md). For generating queries from such a template, you
can use:

```bash
python3 query_gen/gen_queries.py --query_output_dir qreps --template_dir ./templates/imdb/3a/ -n 10 --user imdb --pwd password --db_name imdb --port 5432
```

### Generating Cardinalities (TODO)

## Future Work
