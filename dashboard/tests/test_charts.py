"""Графики строятся на реальных данных, не падают на пустых срезах, доли сходятся."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import charts as ch
from data import DF, YEARS

YEAR = YEARS[-1]
MOBILITY_YEAR = YEARS[-2]
SCHOOL = int(DF[DF['year'] == YEAR]['school_num'].value_counts().index[0])
DISTRICT = DF.loc[DF['school_num'] == SCHOOL, 'district'].iloc[0]
NO_SCHOOL = -1


def _has_data(fig: go.Figure) -> bool:
    def size(t):
        if isinstance(t, go.Table):
            return len(t.cells.values[0]) if t.cells.values else 0
        return 0 if t.x is None else len(t.x)
    return any(size(t) > 0 for t in fig.data)


def _is_stub(fig: go.Figure) -> bool:
    return not fig.data and len(fig.layout.annotations) == 1


@pytest.fixture(scope='module')
def df_forecast():
    """Фиксированные бинарные вероятности только для тестов графиков, не модель."""
    d = DF.copy()
    d['prob_leave'] = np.resize([0.1, 0.3, 0.6, 0.8], len(d))
    return d


SCHOOL_CHARTS = {
    'gaps': lambda df, s: ch.fig_gaps_vs_moscow(df, YEAR, s),
    'age_vs_moscow': lambda df, s: ch.fig_age_school_vs_moscow(df, YEAR, s),
    'categories': lambda df, s: ch.fig_qualification_categories(df, YEAR, s),
    'education': lambda df, s: ch.fig_education_profdev(df, YEAR, s),
    'subjects': lambda df, s: ch.fig_teachers_by_subject(df, YEAR, school_num=s),
    'dynamics': lambda df, s: ch.fig_dynamics_years(df, school_num=s),
}


@pytest.mark.parametrize('name', SCHOOL_CHARTS)
def test_school_chart_has_data(name):
    fig = SCHOOL_CHARTS[name](DF, SCHOOL)
    assert isinstance(fig, go.Figure)
    assert _has_data(fig), f'{name}: пустой график для школы {SCHOOL}'


@pytest.mark.parametrize('name', SCHOOL_CHARTS)
def test_school_chart_stub_for_unknown_school(name):
    fig = SCHOOL_CHARTS[name](DF, NO_SCHOOL)
    assert isinstance(fig, go.Figure)
    assert _is_stub(fig)


def test_city_charts_have_data():
    for fig in (ch.fig_age_moscow(DF, YEAR),
                ch.fig_teachers_by_subject(DF, YEAR),
                ch.fig_mobility_actual(DF, year=MOBILITY_YEAR),
                ch.fig_schools_outflow(DF, MOBILITY_YEAR, DISTRICT),
                ch.fig_dynamics_years(DF, district=DISTRICT)):
        assert _has_data(fig), fig.layout.title.text


def test_mobility_unknown_for_last_year():
    assert _is_stub(ch.fig_mobility_actual(DF, year=YEAR))


def test_education_chart_uses_all_metrics():
    fig = ch.fig_education_profdev(DF, YEAR, SCHOOL)
    assert list(fig.data[0].y) == [m[0] for m in ch.EDU_METRICS]


def test_gaps_include_pk():
    fig = ch.fig_gaps_vs_moscow(DF, YEAR, SCHOOL)
    assert 'ПК за последние 3 года' in fig.data[0].y


def test_age_shares_sum_to_100():
    fig = ch.fig_age_school_vs_moscow(DF, YEAR, SCHOOL)
    for trace in fig.data:
        assert sum(trace.y) == pytest.approx(100, abs=0.5)


def test_category_shares_sum_to_100():
    fig = ch.fig_qualification_categories(DF, YEAR, SCHOOL)
    totals = np.sum([trace.x for trace in fig.data], axis=0)
    assert totals == pytest.approx([100] * len(totals), abs=0.5)


def test_dynamics_matches_raw_shares():
    fig = ch.fig_dynamics_years(DF)
    leave = next(t for t in fig.data if t.name == ch.MOVE_LABELS[ch.MOVE_ORDER[1]])
    raw = DF[DF['year'] == MOBILITY_YEAR]['target'].mean() * 100
    assert leave.y[list(leave.x).index(str(MOBILITY_YEAR))] == pytest.approx(raw, abs=0.1)


def test_forecast_charts_stub_without_model():
    assert _is_stub(ch.fig_risk_by_school(DF, YEAR))
    assert _is_stub(ch.fig_forecast_by_subject(DF, YEAR))
    assert _is_stub(ch.fig_shap_importance(None))
    assert _is_stub(ch.fig_shap_explanation(None))
    assert ch.agg_risk_groups(DF, YEAR).empty


def test_forecast_charts_with_predictions(df_forecast):
    assert _has_data(ch.fig_risk_by_school(df_forecast, YEAR))
    assert _has_data(ch.fig_forecast_by_subject(df_forecast, YEAR))
    groups = ch.agg_risk_groups(df_forecast, YEAR, school_num=SCHOOL)
    assert list(groups.columns) == ch.TABLE_COLS and len(groups) > 0
    assert _has_data(ch.fig_risk_groups_table(groups))


def test_shap_charts_with_input():
    shap = pd.DataFrame({'feature': ['стаж', 'возраст'], 'mean_abs_shap': [0.4, 0.3]})
    assert _has_data(ch.fig_shap_importance(shap))
    expl = pd.DataFrame({'factor': ['стаж', 'возраст'], 'value': [4, 29],
                         'contribution': [0.09, -0.05]})
    assert _has_data(ch.fig_shap_explanation(expl))
