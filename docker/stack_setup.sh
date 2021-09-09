#!/bin/sh

createdb -U $POSTGRES_USER stack
wget -O /var/lib/postgresql/so_pg12 https://www.dropbox.com/s/q66sw2k6fnugsse/so_pg12?dl=1
pg_restore -v -d stack -U $POSTGRES_USER /var/lib/postgresql/so_pg12
