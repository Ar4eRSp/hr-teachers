"""Агрегаты для KPI-карточек, плиток округов и таблиц: DataFrame → числа / DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from charts import MIN_GROUP, MOVE_ORDER, add_risk, forecast_rows, subject_label

MOVED, LEFT = MOVE_ORDER


def mobility_year(df: pd.DataFrame, year: int) -> int:
    """Последний год с известным исходом, не позже выбранного."""
    known = df.loc[df['movement_type'].notna(), 'year']
    return int(min(year, known.max())) if len(known) else year


def pct(mask) -> float:
    return float(np.mean(mask) * 100) if len(mask) else np.nan


def staff_kpis(part: pd.DataFrame) -> dict:
    """Показатели кадрового состава среза (один год)."""
    if part.empty:
        return {}
    return {
        'teachers': part['id'].nunique(),
        'age_mean': part['age'].mean(),
        'age_median': part['age'].median(),
        'exp_ped_median': part['experience_age_pedagogic_clean'].median(),
        'exp_all_median': part['experience_age_all_clean'].median(),
        'cat_high': pct(part['category_qual_cat'] == 'высшая'),
        'cat_first': pct(part['category_qual_cat'] == 'первая'),
        'degree': pct(part['academic_degree_cat'] != 'не_найдено'),
        'class_manager': pct(part['class_manager'] == 1),
        'pk_recent': pct(part['pk_recent'] == 1),
        'men': pct(part['sex'] == 1),
        'per_school': part.groupby('school_num')['id'].nunique().mean(),
    }


def mobility_kpis(part: pd.DataFrame) -> dict:
    """Фактическая мобильность среза (год с известным исходом)."""
    known = part[part['movement_type'].notna()]
    if known.empty:
        return {}
    return {
        'moved': pct(known['movement_type'] == MOVED),
        'left': pct(known['movement_type'] == LEFT),
        'stayed': pct(~known['movement_type'].isin(MOVE_ORDER)),
        'moved_n': int((known['movement_type'] == MOVED).sum()),
        'left_n': int((known['movement_type'] == LEFT).sum()),
    }


def risk_kpis(part: pd.DataFrame) -> dict:
    """Прогнозные показатели - только если в данных есть вероятности модели."""
    d = add_risk(forecast_rows(part))
    if d.empty:
        return {}
    high = d['risk_level'] == 'Высокий'
    return {'risk_high': pct(high), 'risk_high_n': int(high.sum()),
            'prob_leave': d['prob_leave'].mean() * 100,
            'prob_remain': (1 - d['prob_leave']).mean() * 100,
            'predicted_n': len(d), 'total_n': len(part)}


def schools_above_city(df: pd.DataFrame, year: int, district: str | None) -> tuple[int, int]:
    """Сколько школ (от MIN_GROUP педагогов) с оттоком выше среднего по Москве: (выше, всего)."""
    part = df[(df['year'] == year) & df['movement_type'].notna()]
    out = part['movement_type'].isin(MOVE_ORDER)
    city = out.mean()
    g = out.groupby(part['school_num'])
    by_school = g.mean()[g.size() >= MIN_GROUP]
    if district:
        schools = part.loc[part['district'] == district, 'school_num'].unique()
        by_school = by_school[by_school.index.isin(schools)]
    return int((by_school > city).sum()), int(len(by_school))


def district_tiles(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, float, str]:
    """Показатель по округам для плиток: прогнозный риск, а без модели - фактический отток.

    Возвращает (таблица district/value/delta, значение по Москве, название показателя).
    """
    d = add_risk(forecast_rows(df[df['year'] == year]))
    if not d.empty:
        flag = d['risk_level'] == 'Высокий'
        name = 'Высокий риск ухода из системы по округам'
    else:
        y = mobility_year(df, year)
        d = df[(df['year'] == y) & df['movement_type'].notna()]
        flag = d['movement_type'].isin(MOVE_ORDER)
        name = f'Фактический отток по округам за {y}'
    if d.empty:
        return pd.DataFrame(columns=['district', 'value', 'delta']), np.nan, name
    city = flag.mean() * 100
    t = (flag.groupby(d['district']).mean() * 100).rename('value').reset_index()
    t['delta'] = t['value'] - city
    return t.sort_values('district'), city, name


def tile_level(delta: float) -> int:
    """0 - ниже на 1,5+ п.п., 1 - немного ниже, 2 - выше среднего, 3 - выше на 2,5+ п.п."""
    if delta <= -1.5:
        return 0
    if delta < 0:
        return 1
    return 3 if delta >= 2.5 else 2


def school_summary(part: pd.DataFrame, mob_part: pd.DataFrame,
                   year: int, mob_year: int) -> pd.DataFrame:
    """CSV: годы и знаменатели указаны явно, доли малых групп не публикуются."""
    columns = ['school_num', 'district', 'snapshot_year', 'mobility_year',
               'teachers', 'experience_known_n', 'mobility_teachers', 'age_mean',
               'exp_ped_median', 'cat_high_pct', 'pk_recent_pct',
               'students_per_teacher', 'staff_complete', 'moved_pct', 'left_pct']
    if part.empty:
        return pd.DataFrame(columns=columns)
    g = part.groupby(['school_num', 'district']).agg(
        teachers=('id', 'nunique'), age_mean=('age', 'mean'),
        exp_ped_median=('experience_age_pedagogic_clean', 'median'),
        experience_known_n=('experience_age_pedagogic_clean', 'count'),
        cat_high_pct=('category_qual_cat', lambda s: (s == 'высшая').mean() * 100),
        pk_recent_pct=('pk_recent', lambda s: s.mean() * 100),
        students_per_teacher=('students_per_teacher', 'first'),
        staff_complete=('school_staff_complete', 'first'))
    g = g[g['teachers'] >= MIN_GROUP]
    if g.empty:
        return pd.DataFrame(columns=columns)
    mob = mob_part[mob_part['movement_type'].notna()]
    mob_n = mob.groupby(['school_num', 'district'])['id'].nunique()
    g['mobility_teachers'] = mob_n.reindex(g.index).fillna(0).astype(int)
    for col, status in (('moved_pct', MOVED), ('left_pct', LEFT)):
        g[col] = (mob['movement_type'] == status).groupby(
            [mob['school_num'], mob['district']]).mean().where(mob_n >= MIN_GROUP) * 100
    g['snapshot_year'] = year
    g['mobility_year'] = mob_year
    return g.round(1).reset_index()[columns]


SUBJECT_TABLE_COLS = ['Предмет', 'Педагогов', 'Ср. возраст', 'Стаж, медиана',
                      'Высшая кат., %', 'ПК 3 года, %', 'Перешли / ушли']


def subject_table(part: pd.DataFrame, mob_part: pd.DataFrame) -> pd.DataFrame:
    """Таблица "Педагоги по предметам": состав за выбранный год + факт мобильности.

    Группы меньше MIN_GROUP человек не показываются.
    """
    part = part[part['speciality'] != 'не указано']
    if part.empty:
        return pd.DataFrame(columns=SUBJECT_TABLE_COLS)
    g = part.groupby('speciality').agg(
        n=('id', 'nunique'), age=('age', 'mean'),
        exp=('experience_age_pedagogic_clean', 'median'),
        high=('category_qual_cat', lambda s: (s == 'высшая').mean() * 100),
        pk=('pk_recent', lambda s: s.mean() * 100))
    mob = mob_part[mob_part['movement_type'].notna()]
    moved = mob[mob['movement_type'] == MOVED].groupby('speciality')['id'].nunique()
    left = mob[mob['movement_type'] == LEFT].groupby('speciality')['id'].nunique()
    g = g[g['n'] >= MIN_GROUP].sort_values('n', ascending=False)
    values = [
        [subject_label(s) for s in g.index],
        g['n'].astype(int).values,
        g['age'].round(1).values,
        g['exp'].round(0).astype('Int64').values,
        g['high'].round(0).astype(int).values,
        g['pk'].round(0).astype(int).values,
        [f'{int(moved.get(s, 0))} / {int(left.get(s, 0))}' for s in g.index],
    ]
    return pd.DataFrame(dict(zip(SUBJECT_TABLE_COLS, values)))
