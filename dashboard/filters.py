"""Фильтры: верхняя панель (год, округ, школа), боковые доп. фильтры и их применение к данным."""
from __future__ import annotations

import pandas as pd
from dash import html, dcc

from charts import subject_label
from data import DF, YEARS, DISTRICTS, POSITIONS, SUBJECTS

ROLES = {
    'city': 'Городской уровень',
    'district': 'Окружной уровень',
    'director': 'Директор школы',
}

AGE_RANGE = (int(DF['age'].min()), int(DF['age'].max()))
EXP_RANGE = (0, int(DF['experience_age_pedagogic_clean'].max()))
CATEGORIES = {'высшая': 'Высшая', 'первая': 'Первая', 'вторая': 'Вторая',
              'не_найдено': 'Без категории'}

ALL = 'all'
YES_NO = [{'label': 'Все', 'value': ALL}, {'label': 'Да', 'value': 1}, {'label': 'Нет', 'value': 0}]

# id фильтра → значение по умолчанию (по нему же считаются "активные" фильтры и сброс)
EXTRA_DEFAULTS = {
    'f-position': [],
    'f-subject': [],
    'f-sex': ALL,
    'f-age': list(AGE_RANGE),
    'f-exp': list(EXP_RANGE),
    'f-category': [],
    'f-degree': ALL,
    'f-edu-ped': ALL,
    'f-edu-match': ALL,
    'f-pk': ALL,
    'f-pp': ALL,
    'f-class': ALL,
    'f-staff': ['complete'],
}
EXTRA_IDS = list(EXTRA_DEFAULTS)

# фильтр "да/нет" → колонка данных
BINARY_COLUMNS = {
    'f-degree': None,           # учёная степень - отдельная логика
    'f-edu-ped': 'edu_pedagogical',
    'f-edu-match': 'edu_speciality_match',
    'f-pk': 'pk_recent',
    'f-pp': 'retraining',
    'f-class': 'class_manager',
}


def apply_filters(df: pd.DataFrame, values: dict, school: int | None = None) -> pd.DataFrame:
    """Применяет доп. фильтры ко всем годам и всему городу.

    Сравнение "школа vs Москва" остаётся честным: фильтр "Математика" сужает и школу, и город.
    Выбранную школу не отсекает фильтр "только полный штат".
    """
    v = {**EXTRA_DEFAULTS, **{k: x for k, x in values.items() if x is not None}}
    m = pd.Series(True, index=df.index)
    if v['f-position']:
        m &= df['main_position_cat'].isin(v['f-position'])
    if v['f-subject']:
        m &= df['speciality'].isin(v['f-subject'])
    if v['f-sex'] != ALL:
        m &= df['sex'] == int(v['f-sex'])
    m &= df['age'].between(*v['f-age'])
    if list(v['f-exp']) != list(EXP_RANGE):
        m &= df['experience_age_pedagogic_clean'].between(*v['f-exp'])
    if v['f-category']:
        m &= df['category_qual_cat'].isin(v['f-category'])
    if v['f-degree'] != ALL:
        has = df['academic_degree_cat'] != 'не_найдено'
        m &= has if int(v['f-degree']) else ~has
    for fid, col in BINARY_COLUMNS.items():
        if col and v[fid] != ALL:
            m &= df[col] == int(v[fid])
    if 'complete' in v['f-staff']:
        m &= (df['school_staff_complete'] == 1) | (df['school_num'] == school)
    return df[m]


def active_count(values: dict) -> int:
    return sum(1 for k, d in EXTRA_DEFAULTS.items()
               if values.get(k) is not None and values.get(k) != d and k != 'f-staff')


def school_options(district: str | None) -> list[dict]:
    part = DF if not district else DF[DF['district'] == district]
    by_school = part.drop_duplicates('school_num').sort_values('school_num')
    return [{'label': f'Школа № {s}' + ('' if c else ' · неполные данные'), 'value': int(s)}
            for s, c in zip(by_school['school_num'], by_school['school_staff_complete'])]


# ---------------------------------------------------------------------------
# Разметка
# ---------------------------------------------------------------------------
def _labelled(values):
    return [{'label': subject_label(v), 'value': v} for v in values]


def _field(label, control):
    return html.Div([html.Label(label, className='field-label'), control], className='field')


def _radio(fid, options):
    return dcc.RadioItems(id=fid, options=options, value=EXTRA_DEFAULTS[fid],
                          className='segmented', inline=True)


def _group(title, children, open_=False):
    return html.Details([html.Summary(title)] + children, open=open_, className='filter-group')


def top_bar():
    return html.Div([
        _field('Год', dcc.Dropdown(id='f-year', options=YEARS, value=YEARS[-1],
                                   clearable=False, className='dd dd-year')),
        _field('Округ', dcc.Dropdown(id='f-district', options=DISTRICTS, value=None,
                                     placeholder='Все округа', className='dd')),
        _field('Школа', dcc.Dropdown(id='f-school', options=school_options(None), value=None,
                                     placeholder='Все школы', className='dd')),
        html.Div(id='active-chips', className='chips'),
        html.Div([
            html.Button('Сбросить', id='btn-reset', className='btn'),
            html.Button('Выгрузить CSV', id='btn-csv', className='btn btn-primary'),
            dcc.Download(id='download-csv'),
        ], className='bar-actions'),
    ], className='filter-bar')


def sidebar():
    return html.Aside([
        html.Div([html.Span('Дополнительные фильтры'),
                  html.Span(id='filters-badge', className='badge')], className='sidebar-title'),
        _group('Должность и предмет', [
            _field('Основная должность', dcc.Dropdown(
                id='f-position', options=_labelled(POSITIONS), value=[], multi=True,
                placeholder='Все')),
            _field('Предмет', dcc.Dropdown(
                id='f-subject', options=_labelled(SUBJECTS), value=[], multi=True,
                placeholder='Все')),
        ], open_=True),
        _group('Демография', [
            _field('Пол', _radio('f-sex', [{'label': 'Все', 'value': ALL},
                                           {'label': 'Ж', 'value': 0}, {'label': 'М', 'value': 1}])),
            _field('Возраст', dcc.RangeSlider(
                id='f-age', min=AGE_RANGE[0], max=AGE_RANGE[1], step=1, value=list(AGE_RANGE),
                marks=None, tooltip={'placement': 'bottom', 'always_visible': False})),
        ], open_=True),
        _group('Опыт и квалификация', [
            _field('Педагогический стаж, лет', dcc.RangeSlider(
                id='f-exp', min=EXP_RANGE[0], max=EXP_RANGE[1], step=1, value=list(EXP_RANGE),
                marks=None, tooltip={'placement': 'bottom', 'always_visible': False})),
            _field('Квалификационная категория', dcc.Dropdown(
                id='f-category', options=[{'label': v, 'value': k} for k, v in CATEGORIES.items()],
                value=[], multi=True, placeholder='Все')),
            _field('Учёная степень', _radio('f-degree', YES_NO)),
        ]),
        _group('Образование', [
            _field('Высшее педагогическое', _radio('f-edu-ped', YES_NO)),
            _field('Специализация = предмет', _radio('f-edu-match', YES_NO)),
        ]),
        _group('Профразвитие', [
            _field('ПК за последние 3 года', _radio('f-pk', YES_NO)),
            _field('Профпереподготовка', _radio('f-pp', YES_NO)),
        ]),
        _group('Нагрузка и роли', [
            _field('Классное руководство', _radio('f-class', YES_NO)),
        ]),
        _group('Качество данных', [
            dcc.Checklist(id='f-staff', value=EXTRA_DEFAULTS['f-staff'], className='check',
                          options=[{'label': ' Только условно полные выгрузки школ', 'value': 'complete'}]),
            html.P('Полнота оценена по соотношению учеников и педагогов. '
                   'Выбранная школа сохраняется в сравнении. Все показатели учитывают фильтры.',
                   className='hint'),
        ]),
    ], className='sidebar')
