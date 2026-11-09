"""Фильтры, агрегаты для карточек и рендер вкладок дашборда."""
import json

import pandas as pd
import plotly
import pytest

import app as A
import filters as flt
import metrics as mt
from data import DF, YEARS

YEAR = YEARS[-1]
EXTRA = list(flt.EXTRA_DEFAULTS.values())
SCHOOL = A.DEFAULT_SCHOOL
DISTRICT = A.SCHOOL_DISTRICT[SCHOOL]
PARTIAL_SCHOOL = int(DF.loc[DF['school_staff_complete'] == 0, 'school_num'].iloc[0])


# --- фильтры -----------------------------------------------------------------
def test_default_filters_drop_only_partial_schools():
    out = flt.apply_filters(DF, flt.EXTRA_DEFAULTS)
    assert len(out) == (DF['school_staff_complete'] == 1).sum()


def test_selected_partial_school_is_kept():
    out = flt.apply_filters(DF, flt.EXTRA_DEFAULTS, school=PARTIAL_SCHOOL)
    assert (out['school_num'] == PARTIAL_SCHOOL).any()


def test_subject_and_sex_filters():
    out = flt.apply_filters(DF, {'f-subject': ['математики'], 'f-sex': 1, 'f-staff': []})
    assert len(out) > 0
    assert set(out['speciality']) == {'математики'} and set(out['sex']) == {1}


def test_binary_filter_and_ranges():
    out = flt.apply_filters(DF, {'f-pk': 0, 'f-age': [30, 40], 'f-staff': []})
    assert (out['pk_recent'] == 0).all() and out['age'].between(30, 40).all()


def test_active_count_ignores_defaults():
    assert flt.active_count(flt.EXTRA_DEFAULTS) == 0
    assert flt.active_count({**flt.EXTRA_DEFAULTS, 'f-subject': ['химии'], 'f-sex': 0}) == 2


# --- агрегаты ----------------------------------------------------------------
def test_mobility_year_falls_back_to_last_known():
    assert mt.mobility_year(DF, YEAR) == YEARS[-2]
    assert mt.mobility_year(DF, YEARS[0]) == YEARS[0]


def test_mobility_kpis_sum_to_100():
    k = mt.mobility_kpis(DF[DF['year'] == YEARS[-2]])
    assert k['moved'] + k['left'] + k['stayed'] == pytest.approx(100)


def test_district_tiles_without_model_show_actual_outflow():
    table, city, name = mt.district_tiles(DF.assign(prob_leave=float('nan')), YEAR)
    assert 'Фактический отток' in name and len(table) == DF['district'].nunique()
    assert table['delta'].abs().max() < 20 and 0 < city < 100


@pytest.mark.parametrize('delta,level', [(-2, 0), (-0.5, 1), (0, 2), (1, 2), (2.5, 3)])
def test_tile_level(delta, level):
    assert mt.tile_level(delta) == level


def test_subject_table_hides_small_groups():
    part = DF[(DF['year'] == YEAR) & (DF['school_num'] == SCHOOL)]
    table = mt.subject_table(part, DF[(DF['year'] == YEARS[-2]) & (DF['school_num'] == SCHOOL)])
    assert list(table.columns) == mt.SUBJECT_TABLE_COLS
    assert (table['Педагогов'] >= mt.MIN_GROUP).all()


def test_schools_above_city_counts_only_big_schools():
    above, total = mt.schools_above_city(DF, YEARS[-2], None)
    big = DF[DF['year'] == YEARS[-2]].groupby('school_num')['id'].nunique()
    assert 0 < above < total == (big >= mt.MIN_GROUP).sum()


# --- рендер и выгрузка ----------------------------------------------------------
CASES = [('city', None, None), ('district', DISTRICT, None), ('director', DISTRICT, SCHOOL)]


@pytest.mark.parametrize('tab', A.TABS)
@pytest.mark.parametrize('role,district,school', CASES)
@pytest.mark.parametrize('year', YEARS[-2:])
def test_render_every_tab(tab, role, district, school, year):
    out = A.render(tab, year, district, school, role, *EXTRA)
    json.dumps(out, cls=plotly.utils.PlotlyJSONEncoder)  # сериализуется для браузера
    assert out


def test_render_with_empty_selection():
    extra = dict(flt.EXTRA_DEFAULTS, **{'f-age': [16, 17], 'f-exp': [60, 67]})
    for tab in A.TABS:
        assert A.render(tab, YEAR, None, None, 'city', *extra.values())


def test_export_csv_is_school_aggregate():
    out = A.export_csv(1, YEAR, DISTRICT, None, 'district', *EXTRA)
    assert out['filename'].endswith(f'{YEAR}.csv')
    text = out['content'].lstrip('﻿')
    head, *rows = text.strip().splitlines()
    assert 'id' not in head.split(';') and 'school_num' in head
    assert len(rows) == DF[(DF['district'] == DISTRICT) & (DF['school_staff_complete'] == 1)
                           & (DF['year'] == YEAR)]['school_num'].nunique()
