"""Проверки контракта data/teachers.parquet (см. README, "Контракт данных")."""
import pandas as pd
import pytest

import charts as ch
from data import DF, YEARS

REQUIRED = [
    'id', 'year', 'school_num', 'district', 'main_position_cat', 'age', 'age_group',
    'experience_age_pedagogic_clean', 'experience_age_all_clean', 'category_qual_cat',
    'class_manager', 'speciality', 'sex', 'academic_degree_cat',
    'target', 'mobility_status', 'movement_type',
    'n_subjects', 'n_parallels', 'students_per_teacher', 'school_staff_complete',
    'edu_higher', 'edu_pedagogical', 'edu_speciality_match',
    'pk_recent', 'pk_this_year', 'pk_hours_3y', 'retraining',
]
FLAGS = ['class_manager', 'edu_higher', 'edu_pedagogical', 'pk_recent', 'pk_this_year',
         'retraining', 'school_staff_complete']


def test_required_columns():
    missing = set(REQUIRED) - set(DF.columns)
    assert not missing, f'Нет колонок: {sorted(missing)}'


def test_one_row_per_teacher_year():
    assert not DF.duplicated(['id', 'year']).any()


def test_flags_are_binary():
    for c in FLAGS:
        assert set(DF[c].unique()) <= {0, 1}, c
    assert DF['edu_speciality_match'].dropna().isin([0, 1]).all()


def test_experience_quality_flag_matches_known_values():
    columns = ['experience_age_all_clean', 'experience_age_pedagogic_clean', 'experience_age_other_clean']
    valid = DF['experience_valid']
    assert DF.loc[~valid, columns].isna().all().all()
    assert DF.loc[valid, columns].notna().all().all()
    assert (DF.loc[valid, 'experience_age_all_clean'] >= DF.loc[valid, 'experience_age_pedagogic_clean']).all()


def test_mobility_unknown_only_for_last_year():
    last = DF['year'] == YEARS[-1]
    assert DF.loc[last, 'mobility_status'].isna().all()
    assert DF.loc[~last, 'mobility_status'].notna().all()


def test_mobility_status_matches_labels():
    known = DF[DF['mobility_status'].notna()]
    expected = known['mobility_status'].map({0: 'остался', 1: ch.MOVE_ORDER[0], 2: ch.MOVE_ORDER[1]})
    assert (known['movement_type'] == expected).all()


def test_target_is_leave_from_system():
    known = DF[DF['target'].notna()]
    assert ((known['target'] == 1) == (known['mobility_status'] == 2)).all()


def test_with_movement_fallback_matches_data():
    """Запасной расчёт в charts.py даёт то же, что и сборка данных."""
    recomputed = ch.with_movement(DF.drop(columns='movement_type'))['movement_type']
    pd.testing.assert_series_equal(recomputed.fillna('—'), DF['movement_type'].fillna('—'),
                                   check_dtype=False)


def test_teacher_absent_one_year_counts_as_left():
    df = pd.DataFrame({'id': ['a', 'a', 'b', 'b', 'b'],
                       'year': [2022, 2024, 2022, 2023, 2024],
                       'school_num': [1, 1, 1, 2, 2]})
    got = ch.with_movement(df)['movement_type'].tolist()
    assert got[0] == ch.MOVE_ORDER[1]       # a: 2022 → нет в 2023 → ушёл
    assert pd.isna(got[1])                  # a: 2024 - последний год
    assert got[2] == ch.MOVE_ORDER[0]       # b: 2022 → 2023 в другой школе
    assert got[3] == 'остался'              # b: 2023 → 2024 в той же школе


@pytest.mark.parametrize('col', ['students_per_teacher', 'pk_hours_3y', 'age'])
def test_numeric_ranges_are_plausible(col):
    s = DF[col].dropna()
    assert (s >= 0).all()
    if col == 'students_per_teacher':
        assert s.max() <= 60
