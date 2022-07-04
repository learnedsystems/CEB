
# TODO

# New simple workloads

* separate\_sqls.py
* convert\_....py
* get_query_cardinalities.py
* update_qrep_preds.py

# Creating join bitmaps / sample bitmaps

* Use learned-cardinalities repo in /spinning/ ; branch: joincards

For creating join bitmap tables, use:
* scripts/create_sampling_join_bitmaps.py
* bash

```
python3 scripts/create_sampling_tables_bitmaps.py --sample_num 1000
python3 scripts/create_sampling_join_bitmaps.py --sample_num 1000
```

Once tables created, we run queries to update qrep objects with these numbers.
don't want to actually store in the qrep objects because sizes blow up.


* bash gen_joinbitmaps.sh

# Runtime Executions

# Docker setups

* 5431 ---> PG12 before;
* 5433 ---> PG12, after merging the latest pg_hint_plan branch into our repo w/ MAX
card being set
* 5434 ---> PG12, 1GB memory, low-mem params;
* 5435 ---> PG12, 2GB memory, default params , card-db-2gb
* 5436 ---> PG12, 512mb memory, low-mem params, card-db-512mb
* 5437 ---> PG12, full memroy, low-mem params (accidental setup), card-db-1gb

* 5500 ---> ce-benchmark

TODO:
* remove max_card in pg_hint and re-execute
* pg13 version

# joblight training notes

* updated results in the report; check params from there
* bash tmp.sh (TODO: check)

# ceb training, job eval; mscn parameters

* test nh = 4 etc; might help with generalization to JOB?
* hls = 512 seems to help with improving q-error on JOB
* sample size = 10000;
  * would this help?? TODO: test;
  * better sampling might help too;
* bins = 1 vs bins = 10
* eval w indexes disabled; can do it on the 2GB version or full version

# brief result summaries

* port 5435, 2GB RAM; use indexes; JOB avg rts after 2 reps
  Run 1:
  * True: ~8.x
  * Postgres: ~16.x
  * MSCN-joblight2: ~14.x
  * MSCN-ceb-best: ~10.x
  Run 2, after docker restart:
  * True: ~6.6
  * Postgres: ~12.32
  * MSCN-joblight2: ~8.91
  * MSCN-ceb-best: ~10.x


* port 5434; 1GB RAM; use indexes; JOB avg rts after 2 reps
  * True: 10.41
  * Postgres: 23.05
  * MSCN-joblight: 16.33
  * MSCN-joblight2:
  * MSCN-ceb:

* port 5434, run2; 1GB RAM; use indexes; JOB avg rts after 2 reps
  * need clean runs on new aws machine to be sure of the variance here; prev
  run may have been effected by the nn models being trained on 20-30 cpus;
  * maybe w/o restart the low-mem parameters did not have an effect? try to
  re-create instance + run w/o a restart?

  * True: ~6.
  * Postgres: ~6.
  * MSCN-joblight: 8.x (??)
  * MSCN-joblight2:
  * MSCN-ceb:

* port 5437; no indexes; JOB avg rts:
  * True: ~7.33 (!!)

* port 5437; w/ indexes; JOB avg rts: (TODO)
  * True: ~3.5
  * PG: ~5.x

# mscn runs

* baseline avg rts: true: ~8.5; postgres: ~16.5;

Interesting queries: job29, job31, job17, job25
job25 --> low error, but high latency. check true latency here.

* MSCN2239318865 ---> ~2.3 relative cost; low error on 29. TODO: does this
imply faster runtimes on 29 too? qerr = ~450, seems lower than usual?
  * avg rt: 10.66 (!!)

* MSCN1347192799 ---> ~2.6; job29 is high error ---> ~80 second rts too.
  * avg rt: ~12.5-13 seconds

* MSCN.... ---> ~3.0; ~12.5; check;
* MSCN.... ---> ~3.7; ~20; check;

* MSCN-bad-joblight ---> only Q29 is really bad;

# debug restart effect

* create new docker instance, and compare parameters with 5434; ---> shared_mem
etc. might have an effect?
