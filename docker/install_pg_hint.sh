#!/bin/sh
git clone https://github.com/parimarjan/pg_hint_plan.git
cd pg_hint_plan
git fetch
git checkout update
#git checkout updatePG13
make
make install
