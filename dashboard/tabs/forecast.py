"""Бинарный прогноз отсутствия педагога в следующем срезе системы."""
from dash import html

import charts as ch
import components as cm
import metrics as mt
import data
from tabs import Selection


def render(sel: Selection):
    risk = mt.risk_kpis(sel.unit)
    head = [cm.note('Бинарный прогноз: отсутствие в следующем годовом срезе системы. '
                    'Оставшиеся в системе могут работать в той же или другой школе.')]
    if not risk:
        head.append(cm.note(data.MODEL_STATUS or 'Для выбранного года и группы прогноз не рассчитан.'))
    district = sel.district
    if sel.school is not None and not sel.unit.empty:
        district = sel.unit['district'].iloc[0]
    groups = ch.agg_risk_groups(sel.df, sel.year, sel.school)
    return head + [
        cm.kpi_row([
            cm.kpi('Останутся в системе', cm.fmt(risk.get('prob_remain'), 1, '%')),
            cm.kpi('Уйдут из системы', cm.fmt(risk.get('prob_leave'), 1, '%'), accent='orange'),
            cm.kpi('Высокий риск ухода', cm.fmt(risk.get('risk_high_n'), 0),
                   f'{cm.fmt(risk.get("risk_high"), 1, "%")} педагогов' if risk else None,
                   accent='orange'),
            cm.kpi('С прогнозом', cm.fmt(risk.get('predicted_n', 0), 0),
                   f'из {len(sel.unit):,} педагогов'.replace(',', ' ')),
        ]),
        html.Div([
            html.Div(cm.graph(ch.fig_risk_by_school(sel.df, sel.year, district)), className='span-6'),
            html.Div(cm.graph(ch.fig_forecast_by_subject(sel.df, sel.year, sel.school)),
                     className='span-6'),
        ], className='grid'),
        html.Div([
            html.Div(cm.graph(ch.fig_risk_groups_table(groups)), className='span-12'),
        ], className='grid'),
    ]
