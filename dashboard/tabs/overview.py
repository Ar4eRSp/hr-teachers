"""Вкладка "Обзор": сводка по выбранному срезу и сравнение округов."""
from dash import html

import charts as ch
import components as cm
import metrics as mt
from tabs import Selection


def render(sel: Selection):
    unit = sel.unit
    k = mt.staff_kpis(unit)
    my = mt.mobility_year(sel.df, sel.year)
    unit_my = Selection(sel.df, my, sel.district, sel.school, sel.role).unit
    mob = mt.mobility_kpis(unit_my)
    risk = mt.risk_kpis(unit)
    if not k:
        return [cm.note('Нет педагогов для выбранных фильтров')]

    cards = cm.kpi_row([
        cm.kpi('Педагогов', cm.fmt(k['teachers'], 0), f'уникальных, {sel.year}'),
        cm.kpi('Средний возраст', cm.fmt(k['age_mean']), f'медиана {cm.fmt(k["age_median"], 0)} лет'),
        cm.kpi('Пед. стаж, медиана', f'{cm.fmt(k["exp_ped_median"], 0)} лет',
               f'общий стаж — {cm.fmt(k["exp_all_median"], 0)} лет'),
        cm.kpi('Высшая категория', cm.fmt(k['cat_high'], 0, '%'),
               f'первая — {cm.fmt(k["cat_first"], 0, "%")}'),
        cm.kpi(f'Ушли из системы, {my}', cm.fmt(mob.get('left'), 1, '%'),
               f'перешли — {cm.fmt(mob.get("moved"), 1, "%")}', accent='orange'),
        cm.kpi('Высокий риск ухода (прогноз)', cm.fmt(risk.get('risk_high'), 1, '%'),
               f'{cm.fmt(risk["risk_high_n"], 0)} педагогов' if risk else 'нет прогноза для выбранного среза'),
    ])
    return [
        cards,
        html.Div([
            html.Div(cm.district_tiles(sel.df, sel.year, sel.district), className='span-7'),
            html.Div(cm.graph(ch.fig_mobility_by_district(sel.df, my), 330), className='span-5'),
        ], className='grid'),
        html.Div([
            html.Div(cm.graph(ch.fig_age_moscow(sel.df, sel.year, sel.district, sel.school)),
                     className='span-6'),
            html.Div(cm.graph(ch.fig_teachers_by_subject(sel.df, sel.year, sel.school, top_n=8,
                                                         district=sel.district)),
                     className='span-6'),
        ], className='grid'),
    ]
