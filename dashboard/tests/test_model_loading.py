"""Применение готовой модели: подготовка и обучение остаются в model.ipynb."""
import json

import numpy as np
import pandas as pd
import pytest

import data
import charts as ch
import metrics as mt


def frame():
    return pd.DataFrame({'id': ['b', 'a', 'a'], 'year': [2024, 2023, 2024],
                         'age': [45, 30, 31],
                         'main_position_cat': ['новая должность', 'учитель', 'руководитель'],
                         'target': [np.nan, 0, np.nan]}, index=[7, 3, 8])


def metadata():
    return {'schema_version': 1, 'task': 'binary_leave', 'target': 'target', 'positive_class': 1,
            'features': ['age', 'main_position_cat'], 'categorical_features': ['main_position_cat'],
            'classes': [0, 1], 'training_years': [2022, 2023], 'forecast_year': 2024}


@pytest.fixture(scope='module')
def native_model():
    # Только технический пример; проектная модель и её качество здесь не создаются.
    from catboost import CatBoostClassifier
    X = pd.DataFrame({'age': np.arange(20, 60), 'main_position_cat': ['учитель', 'руководитель'] * 20})
    model = CatBoostClassifier(iterations=3, depth=2, random_seed=42, thread_count=1,
                              cat_features=['main_position_cat'], verbose=False, allow_writing_files=False)
    model.fit(X, [0, 0, 1, 1] * 10)
    return model


def write_artifact(directory, model, meta=None):
    model.save_model(str(directory / 'catboost.cbm'), format='cbm')
    (directory / 'metadata.json').write_text(json.dumps(metadata() if meta is None else meta), encoding='utf-8')


def test_native_roundtrip_uses_parquet_values_and_preserves_rows(tmp_path, native_model):
    write_artifact(tmp_path, native_model)
    source = frame()
    got, status = data.load_predictions(source, tmp_path)
    assert status is None
    expected = native_model.predict_proba(source.loc[[7, 8], ['age', 'main_position_cat']])[:, 1]
    np.testing.assert_allclose(got.loc[[7, 8], 'prob_leave'], expected)
    assert pd.isna(got.loc[3, 'prob_leave'])
    pd.testing.assert_frame_equal(got[source.columns], source)
    assert 'prob_leave' not in source
    assert not {'prob_move', 'prob_stay'} & set(got)


@pytest.mark.parametrize('change', [
    {'task': 'multiclass'}, {'positive_class': 0}, {'classes': [0, 1, 2]}, {'classes': [1, 0]},
    {'features': ['main_position_cat', 'age']}, {'features': ['age', 'age']},
    {'features': ['age', 'target']}, {'features': ['age', 'id']},
    {'categorical_features': []}, {'training_years': [2022, 2024]},
    {'forecast_year': '2024'}, {'schema_version': 2},
])
def test_bad_contract_does_not_break_analytics(tmp_path, native_model, change):
    meta = metadata()
    meta.update(change)
    write_artifact(tmp_path, native_model, meta)
    got, status = data.load_predictions(frame(), tmp_path)
    assert status and got['prob_leave'].isna().all()


@pytest.mark.parametrize('probabilities', [(0.8, 0.8), (-0.1, 1.1), (np.nan, 0.2), (np.inf, 0)])
def test_invalid_probabilities_are_not_shown(tmp_path, native_model, monkeypatch, probabilities):
    from catboost import CatBoostClassifier
    write_artifact(tmp_path, native_model)
    monkeypatch.setattr(CatBoostClassifier, 'predict_proba',
                        lambda self, X: np.tile(probabilities, (len(X), 1)))
    got, status = data.load_predictions(frame(), tmp_path)
    assert status and got['prob_leave'].isna().all()


def test_missing_artifacts_keep_forecast_unavailable(tmp_path):
    source = frame()
    got, status = data.load_predictions(source, tmp_path)
    assert status and got['prob_leave'].isna().all()
    pd.testing.assert_frame_equal(got[source.columns], source)


def test_broken_json_keeps_forecast_unavailable(tmp_path):
    (tmp_path / 'metadata.json').write_text('not JSON', encoding='utf-8')
    got, status = data.load_predictions(frame(), tmp_path)
    assert status and got['prob_leave'].isna().all()


@pytest.mark.parametrize('issue', ['missing_column', 'missing_category', 'duplicate_key', 'known_target'])
def test_invalid_vitrine_is_not_silently_prepared(tmp_path, native_model, issue):
    write_artifact(tmp_path, native_model)
    source = frame()
    if issue == 'missing_column':
        source = source.drop(columns='age')
    elif issue == 'missing_category':
        source.loc[7, 'main_position_cat'] = None
    elif issue == 'duplicate_key':
        source = pd.concat([source, source.iloc[:1]])
    else:
        source.loc[7, 'target'] = 1
    before = source.copy(deep=True)
    got, status = data.load_predictions(source, tmp_path)
    assert status and got['prob_leave'].isna().all()
    pd.testing.assert_frame_equal(source, before)


def test_numerical_nan_is_passed_directly_to_catboost(tmp_path, native_model):
    write_artifact(tmp_path, native_model)
    source = frame().astype({'age': float})
    source.loc[7, 'age'] = np.nan
    got, status = data.load_predictions(source, tmp_path)
    assert status is None and got.loc[[7, 8], 'prob_leave'].notna().all()
    assert pd.isna(got.loc[7, 'age'])


def test_model_for_absent_year_has_no_forecast(tmp_path, native_model):
    meta = metadata()
    meta['forecast_year'] = 2025
    write_artifact(tmp_path, native_model, meta)
    got, status = data.load_predictions(frame(), tmp_path)
    assert status and got['prob_leave'].isna().all()


def forecast_fixture():
    return pd.DataFrame({'id': list('abcdef'), 'year': [2024] * 6,
                         'school_num': [1] * 6, 'district': ['Северный'] * 6,
                         'speciality': ['математики'] * 6,
                         'prob_leave': [0, 0.249, 0.25, 0.499, 0.5, np.nan],
                         'movement_type': ['остался'] * 6})


def test_risk_uses_only_leave_and_does_not_classify_missing_as_low():
    df = forecast_fixture()
    df['prob_move'] = 0.99
    df['risk_level'] = 'Высокий'
    got = ch.add_risk(df)
    assert got['risk_level'].iloc[:5].tolist() == ['Низкий', 'Низкий', 'Средний', 'Средний', 'Высокий']
    assert pd.isna(got['risk_level'].iloc[5])
    kpi = mt.risk_kpis(df)
    assert kpi['risk_high'] == 20
    assert kpi['predicted_n'] == 5 and kpi['total_n'] == 6
    assert kpi['prob_leave'] + kpi['prob_remain'] == pytest.approx(100)
    assert 'prob_move' not in kpi


def test_binary_forecast_graphs_and_table_sum_to_100():
    fig = ch.fig_forecast_by_subject(forecast_fixture(), 2024)
    assert [t.name for t in fig.data] == ['Останутся в системе', 'Уйдут из системы']
    assert sum(float(t.x[0]) for t in fig.data) == pytest.approx(100)
    groups = ch.agg_risk_groups(forecast_fixture(), 2024)
    assert groups['Педагогов'].tolist() == [5]
    assert not any('перейдут' in c for c in groups.columns)


def test_unavailable_year_keeps_actual_tiles_and_empty_forecast():
    df = forecast_fixture()
    df['prob_leave'] = np.nan
    assert mt.risk_kpis(df) == {}
    assert not ch.fig_risk_by_school(df, 2024).data
    assert not ch.fig_forecast_by_subject(df, 2024).data
    assert ch.agg_risk_groups(df, 2024).empty
    _, _, title = mt.district_tiles(df, 2024)
    assert 'Фактический' in title


def test_forecast_tab_has_no_predicted_school_transition(monkeypatch):
    from tabs import Selection, forecast
    from plotly.utils import PlotlyJSONEncoder
    monkeypatch.setattr(data, 'MODEL_STATUS', None)
    result = forecast.render(Selection(forecast_fixture(), 2024, None, None, 'city'))
    text = json.dumps(result, cls=PlotlyJSONEncoder, ensure_ascii=False)
    assert 'Останутся в системе' in text and 'Уйдут из системы' in text
    assert 'Вероятно перейдут' not in text and 'prob_move' not in text
