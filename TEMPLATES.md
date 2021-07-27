# Query Templates

All queries are generated using templates defined in /templates.

## Query Format:

TODO: explain the format used in sql_representation.

## Query Name Format:

Each unique query generated using our templates are identified using the
following format:
  template_number + template_variant + query_num e.g., 1a1, 2b10 etc.

* template number: 1...n. Uniquely identifies the join graph in the queries
* template variant: a...z. For a given join graph, the predicates may still be
on different columns. Different template files are used to generate such
variants. e.g., queries 2a1,2a100, 2b1, 2b100 etc.
* query_num: 1...n


## Adding Templates

TODO: add explanation

## Template Details

Here, we present the base templates used for each query template in our
dataset. The predicate conditions are left unspecified, and are added by the
rules given in the template files.

* 1:
  1a: Generated using files in: templates/toml2b/

  ```sql
  SELECT COUNT(*) FROM title as t,
  kind_type as kt,
  movie_info as mi1,
  info_type as it1,
  movie_info as mi2,
  info_type as it2,
  cast_info as ci,
  role_type as rt,
  name as n
  WHERE
  t.id = ci.movie_id
  AND t.id = mi1.movie_id
  AND t.id = mi2.movie_id
  AND mi1.movie_id = mi2.movie_id
  AND mi1.info_type_id = it1.id
  AND mi2.info_type_id = it2.id
  AND it1.id = '3'
  AND it2.id = '4'
  AND t.kind_id = kt.id
  AND ci.person_id = n.id
  AND ci.role_id = rt.id
  AND mi1.info IN (Xgenre)
  AND mi2.info IN (Xlanguage)
  AND kt.kind IN (Xmovie_kind)
  AND rt.role IN (Xrole)
  AND n.gender IN (Xgender)
  AND t.production_year <= Xprod_year_up
  AND Xprod_year_low < t.production_year
  ```

* 2: Same join graph as 1, but added movie_keyword and keyword tables.
  - 2a: Generated using files in templates/toml2d
  - 2b: Generated using files in templates/toml2d2
  - 2c: Generated using files in templates/toml2dtitle

  ```sql
  SELECT COUNT(*) FROM title as t,
  kind_type as kt,
  info_type as it1,
  movie_info as mi1,
  movie_info as mi2,
  info_type as it2,
  cast_info as ci,
  role_type as rt,
  name as n,
  movie_keyword as mk,
  keyword as k
  WHERE
  t.id = ci.movie_id
  AND t.id = mi1.movie_id
  AND t.id = mi2.movie_id
  AND t.id = mk.movie_id
  AND k.id = mk.keyword_id
  AND mi1.movie_id = mi2.movie_id
  AND mi1.info_type_id = it1.id
  AND mi2.info_type_id = it2.id
  AND (Xit1)
  AND (Xit2)
  AND t.kind_id = kt.id
  AND ci.person_id = n.id
  AND ci.role_id = rt.id
  AND (Xmi1)
  AND (Xmi2)
  AND (Xmovie_kind)
  AND (Xrole)
  AND (Xgender)
  AND (Xprod_year_up)
  AND (Xprod_year_low)
  ```

* 3:
  - 3a: Generated using files in templates/toml4

```sql
SELECT COUNT(*) FROM title as t,
movie_keyword as mk, keyword as k,
movie_companies as mc, company_name as cn,
company_type as ct, kind_type as kt,
cast_info as ci, name as n, role_type as rt
WHERE t.id = mk.movie_id
AND t.id = mc.movie_id
AND t.id = ci.movie_id
AND ci.movie_id = mc.movie_id
AND ci.movie_id = mk.movie_id
AND mk.movie_id = mc.movie_id
AND k.id = mk.keyword_id
AND cn.id = mc.company_id
AND ct.id = mc.company_type_id
AND kt.id = t.kind_id
AND ci.person_id = n.id
AND ci.role_id = rt.id
AND (Xprod_year_up)
AND (Xprod_year_low)
AND (Xkeyword)
AND (Xcompany_country)
AND (Xcompany_type)
AND (Xmovie_kind)
AND (Xrole)
AND (Xgender)
```

* 4:
```sql
SELECT COUNT(*)
FROM
name as n,
aka_name as an,
info_type as it1,
person_info as pi1,
cast_info as ci,
role_type as rt
WHERE
n.id = ci.person_id
AND ci.person_id = pi1.person_id
AND it1.id = pi1.info_type_id
AND n.id = pi1.person_id
AND n.id = an.person_id
AND ci.person_id = an.person_id
AND an.person_id = pi1.person_id
AND rt.id = ci.role_id
AND (Xgender)
AND (Xname)
AND (Xcast_note)
AND (Xrole)
AND (Xit1)
```

* 5

```sql
SELECT COUNT(*)
FROM title as t,
movie_info as mi1,
kind_type as kt,
info_type as it1,
info_type as it3,
info_type as it4,
movie_info_idx as mii1,
movie_info_idx as mii2,
movie_keyword as mk,
keyword as k
WHERE
t.id = mi1.movie_id
AND t.id = mii1.movie_id
AND t.id = mii2.movie_id
AND t.id = mk.movie_id
AND mii2.movie_id = mii1.movie_id
AND mi1.movie_id = mii1.movie_id
AND mk.movie_id = mi1.movie_id
AND mk.keyword_id = k.id
AND mi1.info_type_id = it1.id
AND mii1.info_type_id = it3.id
AND mii2.info_type_id = it4.id
AND t.kind_id = kt.id
AND (Xmovie_kind)
AND (Xprod_year_up)
AND (Xprod_year_low)
AND (Xmi1)
AND (Xit1)
AND it3.id = '100'
AND it4.id = '101'
AND (Xrating_up)
AND (Xrating_down)
AND (Xvotes_up)
AND (Xvotes_down)
```

* 6

```sql
SELECT COUNT(*)
FROM title as t,
movie_info as mi1,
kind_type as kt,
info_type as it1,
info_type as it3,
info_type as it4,
movie_info_idx as mii1,
movie_info_idx as mii2,
aka_name as an,
name as n,
info_type as it5,
person_info as pi1,
cast_info as ci,
role_type as rt
WHERE
t.id = mi1.movie_id
AND t.id = ci.movie_id
AND t.id = mii1.movie_id
AND t.id = mii2.movie_id
AND mii2.movie_id = mii1.movie_id
AND mi1.movie_id = mii1.movie_id
AND mi1.info_type_id = it1.id
AND mii1.info_type_id = it3.id
AND mii2.info_type_id = it4.id
AND t.kind_id = kt.id
AND (Xmovie_kind)
AND (Xprod_year_up)
AND (Xprod_year_low)
AND (Xmi1)
AND (Xit1)
AND it3.id = '100'
AND it4.id = '101'
AND (Xrating_up)
AND (Xrating_down)
AND (Xvotes_up)
AND (Xvotes_down)
AND n.id = ci.person_id
AND ci.person_id = pi1.person_id
AND it5.id = pi1.info_type_id
AND n.id = pi1.person_id
AND n.id = an.person_id
AND ci.person_id = an.person_id
AND an.person_id = pi1.person_id
AND rt.id = ci.role_id
AND (Xgender)
AND (Xname)
AND (Xcast_note)
AND (Xrole)
AND (Xit5)
```

* 7

```sql
SELECT COUNT(*)
FROM title as t,
movie_info as mi1,
kind_type as kt,
info_type as it1,
info_type as it3,
info_type as it4,
movie_info_idx as mii1,
movie_info_idx as mii2,
movie_keyword as mk,
keyword as k,
aka_name as an,
name as n,
info_type as it5,
person_info as pi1,
cast_info as ci,
role_type as rt
WHERE
t.id = mi1.movie_id
AND t.id = ci.movie_id
AND t.id = mii1.movie_id
AND t.id = mii2.movie_id
AND t.id = mk.movie_id
AND mk.keyword_id = k.id
AND mi1.info_type_id = it1.id
AND mii1.info_type_id = it3.id
AND mii2.info_type_id = it4.id
AND t.kind_id = kt.id
AND (Xmovie_kind)
AND (Xprod_year_up)
AND (Xprod_year_low)
AND (Xmi1)
AND (Xit1)
AND it3.id = '100'
AND it4.id = '101'
AND (Xrating_up)
AND (Xrating_down)
AND (Xvotes_up)
AND (Xvotes_down)
AND n.id = ci.person_id
AND ci.person_id = pi1.person_id
AND it5.id = pi1.info_type_id
AND n.id = pi1.person_id
AND n.id = an.person_id
AND rt.id = ci.role_id
AND (Xgender)
AND (Xname)
AND (Xcast_note)
AND (Xrole)
AND (Xit5)
```

* 8

```sql
SELECT COUNT(*) FROM title as t,
kind_type as kt,
info_type as it1,
movie_info as mi1,
cast_info as ci,
role_type as rt,
name as n,
movie_keyword as mk,
keyword as k,
movie_companies as mc,
company_type as ct,
company_name as cn
WHERE
t.id = ci.movie_id
AND t.id = mc.movie_id
AND t.id = mi1.movie_id
AND t.id = mk.movie_id
AND mc.company_type_id = ct.id
AND mc.company_id = cn.id
AND k.id = mk.keyword_id
AND mi1.info_type_id = it1.id
AND t.kind_id = kt.id
AND ci.person_id = n.id
AND ci.role_id = rt.id
AND (Xit1)
AND (Xmi)
AND (Xmovie_kind)
AND (Xrole)
AND (Xgender)
AND (Xname)
AND (Xprod_year_up)
AND (Xprod_year_low)
AND (Xcompany_name)
AND (Xcompany_type)
```
