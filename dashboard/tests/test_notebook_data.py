"""Короткие проверки расчётов, которые теперь находятся в ноутбуке сборки."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
notebook = json.loads((ROOT / 'notebooks/Предобработка и очистка.ipynb').read_text(encoding='utf-8'))
source = next(''.join(c['source']) for c in notebook['cells']
              if 'dataset-functions' in c.get('metadata', {}).get('tags', []))
functions = {}
exec(compile(source, 'notebook dataset functions', 'exec'), functions)


def education(year=2022, done=2022, known=True):
    return pd.DataFrame({
        'id': ['a'], 'year': [year], 'type_education': ['Дополнительное образование'],
        'qualification': ['не указано'], 'specialization_edu': ['не указано'],
        'has_specialization': [False], 'has_qualification': [False],
        'date_pass': [done], 'date_known': [known], 'Дата исправлена': [False],
    })


def courses():
    return pd.DataFrame(columns=['id', 'year', 'type_prog', 'hours_tot'])


def test_placeholder_is_not_qualification():
    s = pd.Series([None, '', '-', 'не указано', 'Нет / не указано', 'Учитель'])
    assert functions['has_information'](s).tolist() == [False, False, False, False, False, True]
    got = functions['add_profdev'](pd.DataFrame({'id': ['a'], 'year': [2022]}), education(), courses())
    assert got['retraining'].tolist() == [0]


def test_future_snapshot_does_not_supply_past_profdev():
    t = pd.DataFrame({'id': ['a', 'a'], 'year': [2022, 2024]})
    got = functions['add_profdev'](t, education(year=2024, done=2022), courses())
    assert got['pk_recent'].tolist() == [0, 1]


def test_imputed_date_is_not_evidence_of_recent_course():
    got = functions['add_profdev'](pd.DataFrame({'id': ['a'], 'year': [2022]}),
                                  education(known=False), courses())
    assert got['pk_recent'].tolist() == [0]


def test_corrected_invalid_date_is_not_evidence_of_recent_course():
    edu = education()
    edu['Дата исправлена'] = True
    got = functions['add_profdev'](pd.DataFrame({'id': ['a'], 'year': [2022]}), edu, courses())
    assert got['pk_recent'].tolist() == [0]


def test_known_date_is_kept_when_age_based_check_was_unavailable():
    edu = education()
    edu['Дата исправлена'] = pd.Series([pd.NA], dtype='boolean')
    got = functions['add_profdev'](pd.DataFrame({'id': ['a'], 'year': [2022]}), edu, courses())
    assert got['pk_recent'].tolist() == [1]


def test_conflicting_experience_is_unknown_and_raw_values_are_preserved():
    source = next(''.join(c['source']) for c in notebook['cells']
                  if 'clean-experience' in c.get('metadata', {}).get('tags', []))
    frame = pd.DataFrame({'id': ['a', 'b'], 'year': [2022, 2022], 'age': [40, 40],
                          'experience_age_all': [5, 20], 'experience_age_pedagogic': [10, 15],
                          'experience_age_other': [-5, 5]})
    namespace = {**functions, 'df_clean': frame.copy()}
    exec(compile(source, 'clean experience', 'exec'), namespace)
    result = namespace['df_clean']
    assert result['experience_valid'].tolist() == [False, True]
    columns = ['experience_age_all_clean', 'experience_age_pedagogic_clean', 'experience_age_other_clean']
    assert result.loc[0, columns].isna().all()
    assert result.loc[1, columns].tolist() == [20, 15, 5]
    pd.testing.assert_frame_equal(result[frame.columns], frame)


def test_distinct_courses_are_added_in_full():
    pk = pd.DataFrame({'id': ['a', 'a'], 'year': [2022, 2022],
                       'type_prog': ['ПК', 'ПП'], 'hours_tot': [36, 260]})
    got = functions['add_profdev'](pd.DataFrame({'id': ['a'], 'year': [2022]}),
                                  education(known=False), pk)
    assert got['pk_hours_3y'].tolist() == [296]
    assert got['retraining'].tolist() == [1]


def test_mobility_uses_next_year_and_keeps_last_unknown():
    records = pd.DataFrame({'id': ['a', 'a', 'b', 'b', 'b'],
                            'year': [2022, 2024, 2022, 2023, 2024],
                            'school_num': [1, 1, 1, 2, 2]})
    got = functions['label_mobility'](records)
    assert got['mobility_status'].iloc[[0, 2, 3]].tolist() == [2, 1, 0]
    assert got['target'].iloc[[0, 2, 3]].tolist() == [1, 0, 0]
    assert got.loc[got['year'] == 2024, ['target', 'mobility_status']].isna().all().all()


def test_mobility_rejects_ambiguous_school():
    records = pd.DataFrame({'id': ['a', 'a'], 'year': [2022, 2022], 'school_num': [1, 2]})
    with pytest.raises(AssertionError, match='две школы'):
        functions['label_mobility'](records)
