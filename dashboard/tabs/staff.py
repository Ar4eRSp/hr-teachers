"""Вкладка "Кадровый состав": школа (или округ) против округа и Москвы."""
from dash import html

import charts as ch
import components as cm
import metrics as mt
from tabs import Selection

KPI_FIELDS = [
    # (заголовок, ключ staff_kpis, знаков, суффикс)
    ('Средний возраст', 'age_mean', 1, ''),
    ('Пед. стаж, медиана', 'exp_ped_median', 0, ''),
    ('Высшая категория', 'cat_high', 0, '%'),
    ('С учёной степенью', 'degree', 1, '%'),
    ('Классное руководство', 'class_manager', 0, '%'),
    ('ПК за 3 года', 'pk_recent', 0, '%'),
    ('Мужчин', 'men', 0, '%'),
]


def _compare_kpis(sel: Selection):
    unit = mt.staff_kpis(sel.unit)
    city = mt.staff_kpis(sel.city)
    refs = []
    if sel.school is not None:
        district = sel.unit['district'].iloc[0]
        refs.append(('Округ', mt.staff_kpis(sel.city[sel.city['district'] == district])))
    if sel.school is not None or sel.district:
        refs.append(('Москва', city))

    if sel.school is not None:   # численность школы - против средней школы округа/города
        count = cm.kpi('Педагогов в школе', cm.fmt(unit['teachers'], 0),
                       compare=[(n, cm.fmt(r['per_school'], 0) + ' ср.') for n, r in refs])
    else:
        count = cm.kpi('Педагогов', cm.fmt(unit['teachers'], 0),
                       f'в среднем {cm.fmt(unit["per_school"], 0)} на школу',
                       compare=[(n, cm.fmt(r['per_school'], 0) + ' на шк.') for n, r in refs])
    cards = [count]
    for title, key, digits, suffix in KPI_FIELDS:
        cards.append(cm.kpi(title, cm.fmt(unit[key], digits, suffix),
                            compare=[(n, cm.fmt(r[key], digits, suffix)) for n, r in refs]))
    return cards


def render(sel: Selection):
    if sel.unit.empty:
        return [cm.note('Нет педагогов для выбранных фильтров')]
    my = mt.mobility_year(sel.df, sel.year)
    unit_my = Selection(sel.df, my, sel.district, sel.school, sel.role).unit
    args = (sel.df, sel.year, sel.school, sel.district)
    hint = [] if (sel.school is not None or sel.district) else [
        cm.note('Выберите округ или школу, чтобы сравнить их с Москвой.')]
    missing = int(sel.unit['experience_age_pedagogic_clean'].isna().sum())
    if missing:
        hint.append(cm.note(f'Стаж не определён у {missing} из {len(sel.unit)} педагогов. '
                            'Медианы и распределение стажа рассчитаны по известным значениям.'))
    cards = _compare_kpis(sel)
    return hint + [
        html.Div([
            html.Div(cm.graph(ch.fig_gaps_vs_moscow(*args), 300), className='span-4'),
            html.Div(cm.kpi_row(cards, 4), className='span-8'),
        ], className='grid'),
        html.Div([
            html.Div(cm.graph(ch.fig_age_school_vs_moscow(*args), 320), className='span-4'),
            html.Div(cm.graph(ch.fig_qualification_categories(*args), 320), className='span-4'),
            html.Div(cm.graph(ch.fig_education_profdev(*args), 320), className='span-4'),
        ], className='grid'),
        html.Div([
            html.Div(cm.graph(ch.fig_experience_vs_moscow(*args), 320), className='span-4'),
            html.Div(cm.data_table(
                mt.subject_table(sel.unit, unit_my), 'Педагоги по предметам',
                f'{sel.unit_name}, {sel.year} · сортировка по столбцам · '
                f'перешли / ушли — факт за {my} · группы меньше {mt.MIN_GROUP} человек скрыты',
                'subject-table'), className='span-8'),
        ], className='grid'),
    ]
