from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Стиль (референс: светлый карточный дашборд)
# ---------------------------------------------------------------------------
FONT       = 'Inter, "Segoe UI", system-ui, sans-serif'
CARD_BG    = '#fbfaf8'
INK        = '#1a1a1a'
MUTED      = '#767676'

BLUE_DARK  = '#17365e'   # высшая категория / тёмный акцент
BLUE_MAIN  = '#2f6db4'   # школа / первая категория / переходы
BLUE_LIGHT = '#8fb3d9'   # вторая категория / обзор по Москве
BLUE_PALE  = '#dfe8f2'   # без категории / фон Москвы
ORANGE     = '#c2571b'   # отклонения, отток, высокий риск
ORANGE_LT  = '#e8b48d'   # средний риск

AGE_ORDER  = ['до 25', '25–34', '35–44', '45–54', '55–64', '65+']

CAT_ORDER  = ['высшая', 'первая', 'вторая', 'не_найдено']
CAT_LABELS = {'высшая': 'Высшая', 'первая': 'Первая',
              'вторая': 'Вторая', 'не_найдено': 'Без категории'}
CAT_COLORS = [BLUE_DARK, BLUE_MAIN, BLUE_LIGHT, BLUE_PALE]

MOVE_ORDER  = ['перешёл в другую школу', 'ушёл из системы']
MOVE_LABELS = {'перешёл в другую школу': 'Перешли в другую школу',
               'ушёл из системы': 'Ушли из системы'}
MOVE_COLORS = [BLUE_MAIN, ORANGE]

RISK_ORDER  = ['Высокий', 'Средний', 'Низкий']
RISK_COLORS = [ORANGE, ORANGE_LT, BLUE_PALE]
RISK_MID, RISK_HIGH = 0.25, 0.50
TABLE_COLS = ['Группа', 'Педагогов', 'P(останутся в системе)', 'P(уйдут из системы)']

MIN_GROUP = 5   # группы меньше не показываем: доли шумные, и это почти персональные данные


# ---------------------------------------------------------------------------
# Служебные хелперы
# ---------------------------------------------------------------------------
def _school_label(s) -> str:
    return f'Школа № {s}'


def subject_label(s: str) -> str:
    """"начальных классов" → "Начальных классов" (в данных направление в родительном падеже)."""
    return s[:1].upper() + s[1:] if s else s


def _unit(df: pd.DataFrame, year: int, school_num: int | None = None,
          district: str | None = None):
    """Срез выбранной единицы (школа или округ), весь город за год и подпись единицы.

    Без школы и округа единица пустая, подпись None - графики показывают только Москву.
    """
    city = df[df['year'] == year]
    if school_num is not None:
        return city[city['school_num'] == school_num], city, 'Школа'
    if district:
        return city[city['district'] == district], city, district
    return city.iloc[0:0], city, None


def _outflow_share(part: pd.DataFrame) -> float:
    """Доля перешедших + ушедших, % (только строки с известным исходом)."""
    known = part[part['movement_type'].notna()]
    return known['movement_type'].isin(MOVE_ORDER).mean() * 100 if len(known) else np.nan


def _empty_fig(title: str, msg: str = 'Нет данных для выбранных фильтров') -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, xref='paper', yref='paper',
                       x=0.5, y=0.5, font=dict(size=14, color=MUTED))
    fig = _style(fig, title)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def _style(fig: go.Figure, title: str, subtitle: str | None = None,
           legend: bool = False, horizontal: bool = False) -> go.Figure:
    """Приводит фигуру к единому карточному виду."""
    top = 96 if (legend and subtitle) else 80 if (legend or subtitle) else 56
    
    fig.update_layout(
        template='plotly_white', paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(family=FONT, color=INK, size=12),
        title=dict(text=(f'<b>{title}</b>' +
                         (f'<br><span style="font-size:12px;color:{MUTED}">{subtitle}</span>'
                          if subtitle else '')),
                   x=0, xanchor='left', y=1, yref='container', yanchor='top',
                   pad=dict(t=6), font=dict(family=FONT, size=15, color=INK)),
        margin=dict(l=6, r=28, t=top, b=6),
        bargap=0.35, bargroupgap=0.12,
        showlegend=legend,
        # легенда - в верхнем поле под заголовком, не поверх столбцов
        legend=dict(orientation='h', x=0, xref='container', y=1, yanchor='bottom',
                    bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        # дашборд рисует заголовок в HTML-шапке карточки (components.graph)
        meta=dict(title=title, subtitle=subtitle, legend=legend),
    )
    
    # ИСПРАВЛЕНИЕ: используем showline=False вместо line=dict(width=0)
    # Это стандартный способ скрыть оси, который работает во всех версиях Plotly.
    if horizontal:
        fig.update_yaxes(autorange='reversed', showgrid=False, zeroline=False,
                         showline=False, ticks='', automargin=True)
        fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    else:
        fig.update_xaxes(showgrid=False, zeroline=False, showline=False, 
                         ticks='', automargin=True)
        fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
        
    return fig


def with_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет movement_type, если его нет в данных (обычно его сохраняет ноутбук очистки).

    Исход года Y определяется по году Y+1: остался / перешёл в другую школу / ушёл из системы;
    для последнего года - неизвестен (NaN).
    """
    if 'movement_type' in df.columns:
        return df

    nxt = df[['id', 'year', 'school_num']].assign(year=df['year'] - 1)
    next_school = (df[['id', 'year']]
                   .merge(nxt.rename(columns={'school_num': 'next_school'}),
                          on=['id', 'year'], how='left')['next_school']
                   .set_axis(df.index))
    d = df.copy()
    d['movement_type'] = np.select(
        [d['year'] == d['year'].max(), next_school.isna(), next_school != d['school_num']],
        [None, MOVE_ORDER[1], MOVE_ORDER[0]], default='остался')
    d.loc[d['year'] == d['year'].max(), 'movement_type'] = np.nan
    return d

def forecast_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Только строки с доступной корректной вероятностью бинарного ухода."""
    if 'prob_leave' not in df.columns:
        return df.iloc[:0].copy()
    p = pd.to_numeric(df['prob_leave'], errors='coerce')
    valid = p.between(0, 1)
    d = df.loc[valid].copy()
    d['prob_leave'] = p.loc[valid].astype(float)
    return d


def add_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Бинарный риск ухода; отсутствие прогноза не является низким риском."""
    if 'prob_leave' not in df.columns:
        return df
    p = pd.to_numeric(df['prob_leave'], errors='coerce')
    d = df.copy()
    d['risk_level'] = pd.Series(pd.NA, index=d.index, dtype='string')
    valid = p.between(0, 1)
    d.loc[valid, 'risk_level'] = np.select([p.loc[valid] >= RISK_HIGH, p.loc[valid] >= RISK_MID],
                                         ['Высокий', 'Средний'], default='Низкий')
    return d


# ---------------------------------------------------------------------------
# 1 и 9. Мобильность / школы округа с наибольшим оттоком (horizontal stacked)
# ---------------------------------------------------------------------------
def _outflow_stack(df: pd.DataFrame, year: int, mask, top_n: int | None,
                   title: str, subtitle: str, by: str = 'school_num',
                   label=_school_label, total_row: str | None = None,
                   min_n: int = MIN_GROUP) -> go.Figure:
    d = with_movement(df)
    d = d[mask.loc[d.index]]
    part = d[d['year'] == year]
    if part.empty or part['movement_type'].notna().sum() == 0:
        return _empty_fig(title, f'Исходы за {year} год ещё неизвестны')
    tot = part.groupby(by)['id'].nunique()
    tot = tot[tot >= min_n]   # в маленьких группах доля шумная (1 из 1 = 100%)
    if tot.empty:
        return _empty_fig(title, f'Нет групп от {min_n} педагогов')
    mov = (part[part['movement_type'].notna()]
           .groupby([by, 'movement_type'])['id'].nunique()
           .unstack(fill_value=0).reindex(columns=MOVE_ORDER, fill_value=0))
    share = (mov.reindex(tot.index, fill_value=0).div(tot, axis=0) * 100)
    order = share.sum(axis=1).sort_values(ascending=False)
    share = share.loc[order.index[:top_n] if top_n else order.index]
    labels = [label(s) for s in share.index]

    city = with_movement(df)
    city_share = _outflow_share(city[city['year'] == year])
    if total_row:  # итоговая строка (например, "Москва") внизу списка
        city_part = city[city['year'] == year]
        n = city_part['id'].nunique()
        row = [city_part.loc[city_part['movement_type'] == m, 'id'].nunique() / n * 100
               for m in MOVE_ORDER]
        share.loc[total_row] = row
        labels.append(f'<b>{total_row}</b>')

    fig = go.Figure()
    for m, col in zip(MOVE_ORDER, MOVE_COLORS):
        fig.add_trace(go.Bar(
            name=MOVE_LABELS[m], y=labels,
            x=share[m].round(1), orientation='h',
            marker=dict(color=col, cornerradius=3),
            texttemplate='%{value:.0f}%', textposition='auto',
            insidetextfont=dict(color='white'), outsidetextfont=dict(color=INK),
            hovertemplate='%{y}<br>' + MOVE_LABELS[m] + ': %{value:.1f}%<extra></extra>'))
    fig.update_layout(barmode='stack')
    if not np.isnan(city_share) and not total_row:
        fig.add_vline(x=city_share, line_width=1.5, line_color=INK)
        subtitle += f' {city_share:.1f}%'.replace('.', ',')
    return _style(fig, title, subtitle, legend=True, horizontal=True)


def fig_mobility_actual(df, year: int = 2024, district: str | None = None,
                        top_n: int = 10) -> go.Figure:
    mask = (df['district'] == district) if district else pd.Series(True, index=df.index)
    return _outflow_stack(df, year, mask, top_n,
                          f'Фактическая мобильность за {year}',
                          'Доля педагогов, перешедших или ушедших из системы')


def fig_schools_outflow(df, year: int, district: str | None, top_n: int = 10) -> go.Figure:
    mask = (df['district'] == district) if district else pd.Series(True, index=df.index)
    where = 'округа' if district else 'Москвы'
    return _outflow_stack(df, year, mask, top_n,
                          f'Школы {where} с наибольшим оттоком',
                          f'{district or "Москва"} · перешли + ушли за {year}, % · '
                          f'школы от {MIN_GROUP} педагогов · метка — среднее по Москве')


def fig_mobility_by_district(df, year: int) -> go.Figure:
    return _outflow_stack(df, year, pd.Series(True, index=df.index), None,
                          f'Фактическая мобильность за {year} год',
                          'Доля педагогов, % · по округам', by='district',
                          label=str, total_row='Москва')


# ---------------------------------------------------------------------------
# 2. Возрастная структура - обзор (vertical bar)
# ---------------------------------------------------------------------------
def fig_age_moscow(df, year: int, district: str | None = None,
                   school_num: int | None = None) -> go.Figure:
    part, city, label = _unit(df, year, school_num, district)
    if label is None:
        part = city
    if part.empty:
        return _empty_fig('Возрастная структура')
    vc = part['age_group'].value_counts(normalize=True).reindex(AGE_ORDER, fill_value=0) * 100
    where = {None: 'Москва', 'Школа': f'Школа № {school_num}'}.get(label, label)
    fig = go.Figure(go.Bar(x=AGE_ORDER, y=vc.round(1), marker=dict(color=BLUE_MAIN, cornerradius=3),
                           text=[f'{v:.1f}%' for v in vc], textposition='outside',
                           cliponaxis=False, outsidetextfont=dict(color=INK),
                           hovertemplate='%{x}: %{y:.1f}%<extra>' + where + '</extra>'))
    mean_age = f'{part["age"].mean():.1f}'.replace('.', ',')
    return _style(fig, 'Возрастная структура',
                  f'{where} · доля педагогов, % · средний возраст {mean_age}'
                  f' · медиана {part["age"].median():.0f}')


# ---------------------------------------------------------------------------
# 3. Педагоги по предметам (horizontal bar)
# ---------------------------------------------------------------------------
def fig_teachers_by_subject(df, year: int, school_num: int | None = None,
                            top_n: int = 15, district: str | None = None) -> go.Figure:
    part = df[(df['year'] == year) & (df['speciality'] != 'не указано')]
    if school_num is not None:
        part = part[part['school_num'] == school_num]
    elif district:
        part = part[part['district'] == district]
    if part.empty:
        return _empty_fig('Педагоги по предметам')
    cnt = part.groupby('speciality')['id'].nunique().sort_values(ascending=False).head(top_n)
    fig = go.Figure(go.Bar(y=[subject_label(s) for s in cnt.index], x=cnt.values, orientation='h',
                           marker=dict(color=BLUE_LIGHT, cornerradius=3),
                           text=[f'{v:,}'.replace(',', ' ') for v in cnt.values],
                           textposition='outside', cliponaxis=False,
                           outsidetextfont=dict(color=INK),
                           hovertemplate='%{y}: %{x} педагогов<extra></extra>'))
    return _style(fig, 'Педагоги по предметам', f'Число уникальных педагогов, топ-{top_n}',
                  horizontal=True)


# ---------------------------------------------------------------------------
# 4. Где школа (округ) отстаёт от Москвы (horizontal bar, отклонения в п.п.)
# ---------------------------------------------------------------------------
def _gap_metrics(sch: pd.DataFrame, city: pd.DataFrame):
    out = []
    if 'pk_recent' in sch.columns:
        out.append(('ПК за последние 3 года', sch['pk_recent'].mean() * 100,
                    city['pk_recent'].mean() * 100))
    out += [
        ('Высшая категория', (sch['category_qual_cat'] == 'высшая').mean() * 100,
         (city['category_qual_cat'] == 'высшая').mean() * 100),
        ('Доля педагогов 55+', (sch['age'] >= 55).mean() * 100,
         (city['age'] >= 55).mean() * 100),
        ('Учёная степень', (sch['academic_degree_cat'] != 'не_найдено').mean() * 100,
         (city['academic_degree_cat'] != 'не_найдено').mean() * 100),
    ]
    return out


def fig_gaps_vs_moscow(df, year: int, school_num: int | None = None,
                       district: str | None = None) -> go.Figure:
    sch, city, label = _unit(df, year, school_num, district)
    title = 'Отклонения от среднего по городу'
    if label is None:
        return _empty_fig(title, 'Выберите округ или школу')
    if sch.empty:
        return _empty_fig(title)
    rows = [(n, s - c) for n, s, c in _gap_metrics(sch, city)]
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    labels = [r[0] for r in rows]
    devs = [r[1] for r in rows]
    texts = [f'{v:+.1f}'.replace('-', '−').replace('.', ',') + ' п.п.' for v in devs]
    # длина полосы - модуль отклонения, знак - в подписи (как в макете)
    fig = go.Figure(go.Bar(y=labels, x=[round(abs(v), 1) for v in devs], orientation='h',
                           marker=dict(color=BLUE_MAIN, cornerradius=3),
                           text=texts, textposition='outside', cliponaxis=False,
                           outsidetextfont=dict(color=INK),
                           hovertemplate='%{y}: %{text}<extra></extra>'))
    fig = _style(fig, title, 'Разность, п.п. · знак не означает оценку качества',
                 horizontal=True)
    fig.update_layout(margin=dict(r=70))
    return fig


# ---------------------------------------------------------------------------
# 5. Возрастная структура - школа (округ) vs Москва (grouped vertical bar)
# ---------------------------------------------------------------------------
def fig_age_school_vs_moscow(df, year: int, school_num: int | None = None,
                             district: str | None = None) -> go.Figure:
    sch, city, label = _unit(df, year, school_num, district)
    if label is not None and sch.empty:
        return _empty_fig('Возрастная структура')
    groups = ([(label, sch, BLUE_MAIN)] if label else []) + [('Москва', city, BLUE_PALE)]
    fig = go.Figure()
    for name, sub, col in groups:
        vc = sub['age_group'].value_counts(normalize=True).reindex(AGE_ORDER, fill_value=0) * 100
        fig.add_trace(go.Bar(name=name, x=AGE_ORDER, y=vc.round(1),
                             marker=dict(color=col, cornerradius=3),
                             hovertemplate=name + ' · %{x}: %{value:.1f}%<extra></extra>'))
    fig.update_layout(barmode='group')
    return _style(fig, 'Возрастная структура', 'Доля педагогов, %', legend=True)


# ---------------------------------------------------------------------------
# 5б. Стаж - школа (округ) vs Москва (grouped vertical bar)
# ---------------------------------------------------------------------------
EXP_BINS   = [-1, 3, 10, 20, 30, np.inf]
EXP_ORDER  = ['до 3 лет', '4–10', '11–20', '21–30', '31+']


def fig_experience_vs_moscow(df, year: int, school_num: int | None = None,
                             district: str | None = None) -> go.Figure:
    sch, city, label = _unit(df, year, school_num, district)
    if label is not None and sch.empty:
        return _empty_fig('Педагогический стаж')
    if (not city['experience_age_pedagogic_clean'].notna().any()
            or (label is not None and not sch['experience_age_pedagogic_clean'].notna().any())):
        return _empty_fig('Педагогический стаж', 'Нет достоверных значений стажа для этой группы')
    groups = ([(label, sch, BLUE_MAIN)] if label else []) + [('Москва', city, BLUE_PALE)]
    fig = go.Figure()
    for name, sub, col in groups:
        grp = pd.cut(sub['experience_age_pedagogic_clean'], EXP_BINS, labels=EXP_ORDER)
        vc = grp.value_counts(normalize=True).reindex(EXP_ORDER, fill_value=0) * 100
        fig.add_trace(go.Bar(name=name, x=EXP_ORDER, y=vc.round(1),
                             marker=dict(color=col, cornerradius=3),
                             hovertemplate=name + ' · %{x}: %{value:.1f}%<extra></extra>'))
    fig.update_layout(barmode='group')
    return _style(fig, 'Педагогический стаж',
                  'Доля педагогов с известным стажем, %', legend=True)


# ---------------------------------------------------------------------------
# 6. Квалификационные категории (horizontal 100% stacked)
# ---------------------------------------------------------------------------
def fig_qualification_categories(df, year: int, school_num: int | None = None,
                                 district: str | None = None) -> go.Figure:
    sch, city, label = _unit(df, year, school_num, district)
    if label is not None and sch.empty:
        return _empty_fig('Квалификационные категории')
    rows = []
    if label == 'Школа':
        rows.append(('Школа', sch))
        district = sch['district'].dropna().unique()
        district = district[0] if len(district) else None
    if district:
        rows.append((district, city[city['district'] == district]))
    rows.append(('Москва', city))
    fig = go.Figure()
    for c, col in zip(CAT_ORDER, CAT_COLORS):
        vals = [(sub['category_qual_cat'] == c).mean() * 100 if len(sub) else 0
                for _, sub in rows]
        fig.add_trace(go.Bar(
            name=CAT_LABELS[c], y=[n for n, _ in rows], x=[round(v, 1) for v in vals],
            orientation='h', marker=dict(color=col, cornerradius=2),
            text=[f'{v:.0f}' for v in vals], textposition='inside',
            insidetextfont=dict(color=INK if col == BLUE_PALE else 'white'),
            hovertemplate='%{y} · ' + CAT_LABELS[c] + ': %{value:.1f}%<extra></extra>'))
    fig.update_layout(barmode='stack')
    return _style(fig, 'Квалификационные категории', 'Структура, % от педагогов',
                  legend=True, horizontal=True)


# ---------------------------------------------------------------------------
# 7. Образование и профразвитие (bar = школа/округ, засечка = Москва)
# ---------------------------------------------------------------------------
EDU_METRICS = [
    ('Высшее педагогическое', 'edu_pedagogical', 'pct'),
    ('Специализация = предмет', 'edu_speciality_match', 'pct'),
    ('Переподготовка (ПП)', 'retraining', 'pct'),
    ('Часы ПК за 3 года, ср.', 'pk_hours_3y', 'mean'),
]


def fig_education_profdev(df, year: int, school_num: int | None = None,
                          district: str | None = None) -> go.Figure:
    sch, city, label = _unit(df, year, school_num, district)
    if label is None:
        sch, label = city, 'Москва'
    avail = [m for m in EDU_METRICS if m[1] in city.columns]
    if sch.empty or not avail:
        return _empty_fig('Образование и профразвитие',
                          'Нет колонок образования/ПК (см. контракт данных)')
    labels, sv, cv, texts = [], [], [], []
    for name, col, how in avail:
        f = (lambda s: s.mean() * 100) if how == 'pct' else (lambda s: s.mean())
        labels.append(name); sv.append(f(sch[col])); cv.append(f(city[col]))
        texts.append(f'{sv[-1]:.0f}' if label == 'Москва' else f'{sv[-1]:.0f}  ·  Москва {cv[-1]:.0f}')
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[round(v, 1) for v in sv], orientation='h', name=label,
                         marker=dict(color=BLUE_LIGHT, cornerradius=3),
                         text=texts, textposition='outside', cliponaxis=False,
                         outsidetextfont=dict(color=INK),
                         hovertemplate='%{y}: %{text}<extra></extra>'))
    if label != 'Москва':
        fig.add_trace(go.Scatter(y=labels, x=[round(v, 1) for v in cv], mode='markers',
                                 name='Москва',
                                 marker=dict(symbol='line-ns', size=18, color=INK,
                                             line=dict(width=2.5, color=INK)),
                                 hovertemplate='%{y}: %{x:.0f}<extra>Москва</extra>'))
    subtitle = ('%, часы — в часах · Москва' if label == 'Москва'
                else f'{label} (полоса) и Москва (засечка) · %, часы — в часах')
    fig = _style(fig, 'Образование и профразвитие', subtitle, horizontal=True)
    fig.update_layout(margin=dict(r=120 if label != 'Москва' else 40))
    return fig


# ---------------------------------------------------------------------------
# 8. Динамика по годам (vertical stacked; с единицей - рядом столбцы Москвы)
# ---------------------------------------------------------------------------
def _dynamics_shares(d: pd.DataFrame) -> pd.DataFrame:
    known = d[d['movement_type'].notna()]
    years = sorted(known['year'].unique())
    denom = d[d['year'].isin(years)].groupby('year')['id'].nunique()
    seg = (known.groupby(['year', 'movement_type'])['id'].nunique()
           .unstack(fill_value=0).reindex(columns=MOVE_ORDER, fill_value=0))
    return seg.div(denom, axis=0) * 100


def fig_dynamics_years(df, district: str | None = None,
                       school_num: int | None = None) -> go.Figure:
    city = with_movement(df)
    d = city
    if district:
        d = d[d['district'] == district]
    if school_num is not None:
        d = d[d['school_num'] == school_num]
    if d['movement_type'].notna().sum() == 0:
        return _empty_fig('Динамика по годам')
    unit = (f'Школа № {school_num}' if school_num is not None else district)
    series = [(unit, _dynamics_shares(d), MOVE_COLORS)] if unit else []
    series.append(('Москва', _dynamics_shares(city),
                   [BLUE_LIGHT, ORANGE_LT] if unit else MOVE_COLORS))

    fig = go.Figure()
    for name, share, colors in series:
        x = ([[str(y) for y in share.index], [name] * len(share)] if unit
             else [str(y) for y in share.index])
        for m, col in zip(MOVE_ORDER, colors):
            fig.add_trace(go.Bar(name=MOVE_LABELS[m], x=x, y=share[m].round(1),
                                 legendgroup=m, showlegend=name == series[0][0],
                                 marker=dict(color=col, cornerradius=3),
                                 texttemplate='%{value:.0f}%', textposition='auto',
                                 insidetextfont=dict(color='white'),
                                 outsidetextfont=dict(color=INK),
                                 hovertemplate=name + ' · ' + MOVE_LABELS[m] +
                                               ': %{value:.1f}%<extra></extra>'))
    fig.update_layout(barmode='stack')
    subtitle = f'Доля педагогов, % · {unit} против Москвы' if unit else 'Доля педагогов, % · Москва'
    return _style(fig, 'Динамика по годам', subtitle, legend=True)


# ---------------------------------------------------------------------------
# 8б. Где мобильность выше: предмет × возрастная группа (heatmap)
# ---------------------------------------------------------------------------
HEAT_BINS   = [10, 15, 20]                      # границы: <10, 10–15, 15–20, 20+
HEAT_LABELS = ['< 10%', '10–15%', '15–20%', '20%+']
HEAT_COLORS = ['#eef2f7', BLUE_LIGHT, ORANGE_LT, ORANGE]


def fig_mobility_heatmap(df, year: int, school_num: int | None = None,
                         district: str | None = None, top_n: int = 8,
                         min_n: int = 5) -> go.Figure:
    title = 'Где мобильность выше: предмет × возрастная группа'
    d = with_movement(df)
    d = d[(d['year'] == year) & (d['speciality'] != 'не указано') & d['movement_type'].notna()]
    if school_num is not None:
        d = d[d['school_num'] == school_num]
    elif district:
        d = d[d['district'] == district]
    if d.empty:
        return _empty_fig(title, f'Исходы за {year} год ещё неизвестны')
    subjects = d['speciality'].value_counts().head(top_n).index
    d = d[d['speciality'].isin(subjects)].assign(out=d['movement_type'].isin(MOVE_ORDER))
    g = d.groupby(['speciality', 'age_group'], observed=False)['out'].agg(['mean', 'size'])
    share = (g['mean'] * 100).unstack().reindex(index=subjects, columns=AGE_ORDER)
    n = g['size'].unstack().reindex(index=subjects, columns=AGE_ORDER).fillna(0)
    share = share.where(n >= min_n)

    z = np.digitize(share.fillna(-1).values, HEAT_BINS).astype(float)
    z[share.isna().values] = np.nan
    text = np.where(share.isna(), 'н/д', share.round(0).astype('Int64').astype(str) + '%')
    k = len(HEAT_COLORS)
    scale = [[pos, c] for i, c in enumerate(HEAT_COLORS) for pos in (i / k, (i + 1) / k)]
    fig = go.Figure(go.Heatmap(
        z=z, x=AGE_ORDER, y=[subject_label(s) for s in subjects], text=text, texttemplate='%{text}',
        customdata=n.values, zmin=-0.5, zmax=k - 0.5, colorscale=scale, xgap=3, ygap=3,
        showscale=False, hoverongaps=False,
        hovertemplate='%{y} · %{x}: %{text}<br>педагогов: %{customdata}<extra></extra>'))
    for lbl, col in zip(HEAT_LABELS, HEAT_COLORS):   # легенда цветов
        fig.add_trace(go.Bar(x=[None], y=[None], name=lbl, marker=dict(color=col)))
    fig = _style(fig, title, f'Доля перешедших и ушедших за {year}, % · '
                             f'ячейки меньше {min_n} человек скрыты',
                 legend=True, horizontal=True)
    fig.update_xaxes(visible=True, side='top', showgrid=False, zeroline=False, ticks='')
    return fig


# ---------------------------------------------------------------------------
# 10. Группы риска по школам (horizontal stacked)
# ---------------------------------------------------------------------------
def fig_risk_by_school(df, year: int, district: str | None = None,
                       top_n: int = 10) -> go.Figure:
    d = add_risk(forecast_rows(df[df['year'] == year]))
    if d.empty:
        return _empty_fig('Группы риска ухода по школам', 'Нет прогноза для выбранного года и группы')
    if district:
        d = d[d['district'] == district]
    tot = d.groupby('school_num')['id'].nunique()
    tot = tot[tot >= MIN_GROUP]
    if tot.empty:
        return _empty_fig('Группы риска ухода по школам', f'Нет групп от {MIN_GROUP} педагогов')
    seg = (d.groupby(['school_num', 'risk_level'])['id'].nunique()
           .unstack(fill_value=0).reindex(columns=RISK_ORDER, fill_value=0))
    share = seg.reindex(tot.index).div(tot, axis=0) * 100
    order = share['Высокий'].sort_values(ascending=False).head(top_n).index
    share = share.loc[order]
    fig = go.Figure()
    for r, col in zip(RISK_ORDER, RISK_COLORS):
        fig.add_trace(go.Bar(name=r, y=[f'{_school_label(s)} · N={tot[s]}' for s in share.index],
                             x=share[r].round(1), orientation='h',
                             marker=dict(color=col, cornerradius=3),
                             texttemplate='%{value:.0f}%', textposition='auto',
                             insidetextfont=dict(color='white' if r == 'Высокий' else INK),
                             outsidetextfont=dict(color=INK),
                             hovertemplate='%{y} · ' + r + ': %{value:.1f}%<extra></extra>'))
    fig.update_layout(barmode='stack')
    return _style(fig, 'Группы риска ухода по школам',
                  f'Доля педагогов с прогнозом · группы от {MIN_GROUP} человек после фильтров',
                  legend=True, horizontal=True)


# ---------------------------------------------------------------------------
# 11. Прогнозный статус по предметам (horizontal stacked)
# ---------------------------------------------------------------------------
def fig_forecast_by_subject(df, year: int, school_num: int | None = None,
                            top_n: int = 12) -> go.Figure:
    d = df[(df['year'] == year) & (df['speciality'] != 'не указано')]
    if school_num is not None:
        d = d[d['school_num'] == school_num]
    d = forecast_rows(d)
    if d.empty:
        return _empty_fig('Прогноз ухода по предметам', 'Нет прогноза для выбранного года и группы')
    g = d.groupby('speciality').agg(prob_leave=('prob_leave', 'mean'), n=('id', 'nunique'))
    g = g[g['n'] >= MIN_GROUP].sort_values('prob_leave', ascending=False).head(top_n)
    if g.empty:
        return _empty_fig('Прогноз ухода по предметам', f'Нет групп от {MIN_GROUP} педагогов')
    g['prob_leave'] *= 100
    g['prob_remain'] = 100 - g['prob_leave']
    fig = go.Figure()
    for col, name, color in (('prob_remain', 'Останутся в системе', BLUE_MAIN),
                             ('prob_leave', 'Уйдут из системы', ORANGE)):
        fig.add_trace(go.Bar(name=name, y=[f'{subject_label(s)} · N={int(g.loc[s, "n"])}' for s in g.index],
                             x=g[col].round(1), orientation='h',
                             marker=dict(color=color, cornerradius=3),
                             texttemplate='%{value:.0f}%', textposition='auto',
                             insidetextfont=dict(color='white'),
                             outsidetextfont=dict(color=INK),
                             hovertemplate='%{y} · ' + name +
                                           ': %{value:.1f}%<extra></extra>'))
    fig.update_layout(barmode='stack')
    return _style(fig, 'Прогноз ухода по предметам',
                  f'Средние оценки, % · группы от {MIN_GROUP} человек после фильтров',
                  legend=True, horizontal=True)


# ---------------------------------------------------------------------------
# 12. Что сильнее всего влияет на прогноз (|SHAP|, horizontal bar)
# ---------------------------------------------------------------------------
def fig_shap_importance(shap_df: pd.DataFrame | None, top_n: int = 12) -> go.Figure:
    if shap_df is None or shap_df.empty:
        return _empty_fig('Что сильнее всего влияет на прогноз',
                          'Нет shap_summary.csv (feature, mean_abs_shap)')
    d = shap_df.sort_values('mean_abs_shap', ascending=False).head(top_n)
    fig = go.Figure(go.Bar(y=d['feature'], x=d['mean_abs_shap'].round(3), orientation='h',
                           marker=dict(color=BLUE_MAIN, cornerradius=3),
                           text=d['mean_abs_shap'].round(2), textposition='outside',
                           outsidetextfont=dict(color=INK),
                           hovertemplate='%{y}: %{x:.3f}<extra></extra>'))
    return _style(fig, 'Что сильнее всего влияет на прогноз',
                  'Средний вклад признака по модулю, |SHAP|', horizontal=True)


# ---------------------------------------------------------------------------
# 13. Группы педагогов с наибольшим риском (таблица)
# ---------------------------------------------------------------------------
def agg_risk_groups(df, year: int, school_num: int | None = None, top_n: int = 8) -> pd.DataFrame:
    """Группы по среднему бинарному риску ухода из системы."""
    d = forecast_rows(df[df['year'] == year])
    if school_num is not None:
        d = d[d['school_num'] == school_num]
    if d.empty:
        return pd.DataFrame(columns=TABLE_COLS)
    d = d[d['speciality'] != 'не указано']
    g = (d.groupby('speciality')
          .agg(n=('id', 'nunique'), leave=('prob_leave', 'mean')))
    g = g[g['n'] >= MIN_GROUP].sort_values('leave', ascending=False).head(top_n)
    rows = []
    for subj, r in g.iterrows():
        rows.append([subject_label(subj), int(r['n']), f'{(1 - r["leave"]) * 100:.0f}%',
                     f'{r["leave"] * 100:.0f}%'])
    return pd.DataFrame(rows, columns=TABLE_COLS)


def fig_risk_groups_table(groups_df: pd.DataFrame) -> go.Figure:
    if groups_df is None or groups_df.empty:
        return _empty_fig('Группы педагогов с наибольшим риском', 'Нет прогноза')
    n = len(groups_df)
    fig = go.Figure(go.Table(
        header=dict(values=list(groups_df.columns), fill=dict(color=BLUE_DARK),
                    font=dict(color='white', family=FONT, size=12),
                    align=['left', 'right', 'right', 'right'], height=30),
        cells=dict(values=[groups_df[c].tolist() for c in groups_df.columns],
                   align=['left', 'right', 'right', 'right'],
                   fill=dict(color=['#ffffff' if i % 2 else CARD_BG for i in range(n)]),
                   font=dict(family=FONT, size=12, color=INK), height=28)))
    fig.update_layout(margin=dict(l=6, r=6, t=56, b=6))
    return _style(fig, 'Группы педагогов с наибольшим риском',
                  'Группы по предметам · метрики риска')


# ---------------------------------------------------------------------------
# 14. Почему такой прогноз (двусторонняя диаграмма отклонений)
# ---------------------------------------------------------------------------
def fig_shap_explanation(contrib_df: pd.DataFrame | None) -> go.Figure:
    if contrib_df is None or contrib_df.empty:
        return _empty_fig('Почему такой прогноз',
                          'Нет shap_explanation.csv (factor, value, contribution)')
    d = contrib_df.sort_values('contribution')
    colors = [ORANGE if c > 0 else BLUE_MAIN for c in d['contribution']]
    fig = go.Figure(go.Bar(
        y=[f'{f} = {v}' for f, v in zip(d['factor'], d['value'])],
        x=d['contribution'].round(3), orientation='h',
        marker=dict(color=colors, cornerradius=3),
        text=[f'{c:+.2f}' for c in d['contribution']], textposition='outside',
        cliponaxis=False, outsidetextfont=dict(color=INK),
        hovertemplate='%{y}: %{x:+.3f}<extra></extra>'))
    fig.add_vline(x=0, line_width=1, line_color=INK)
    return _style(fig, 'Почему такой прогноз',
                  'Вклад в риск: влево — снижает, вправо — повышает', horizontal=True)
