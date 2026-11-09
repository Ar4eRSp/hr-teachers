from dash import Dash, html, dcc, Input, Output, State, ALL, ctx, no_update

import filters as flt
from charts import subject_label
import metrics as mt
from data import DF, DISTRICTS, YEARS
from tabs import Selection, overview, staff, mobility, forecast, whatif

TABS = {
    'overview': ('Обзор', overview),
    'staff': ('Кадровый состав', staff),
    'mobility': ('Мобильность', mobility),
    'forecast': ('Прогноз', forecast),
    'whatif': ('What-if', whatif),
}
EXTRA_INPUTS = [Input(fid, 'value') for fid in flt.EXTRA_IDS]

# школа по умолчанию для роли "Директор" - самая крупная за последний год
DEFAULT_SCHOOL = int(DF[DF['year'] == YEARS[-1]]['school_num'].value_counts().index[0])
SCHOOL_DISTRICT = DF.drop_duplicates('school_num').set_index('school_num')['district'].to_dict()

app = Dash(__name__, title='СППР · Кадровая динамика', suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Header([
        html.Div('СППР · Кадровая динамика', className='brand'),
        dcc.Tabs(id='tabs', value='overview', className='nav-tabs',
                 children=[dcc.Tab(label=label, value=key, className='nav-tab',
                                   selected_className='nav-tab--selected')
                           for key, (label, _) in TABS.items()]),
        html.Div([
            html.Span('демо', className='tag', title='Роль выбирается вручную, без авторизации'),
            html.Span('Роль', className='role-label'),
            dcc.Dropdown(id='f-role', value='city', clearable=False, searchable=False,
                         options=[{'label': v, 'value': k} for k, v in flt.ROLES.items()],
                         className='dd dd-role'),
        ], className='header-right'),
    ], className='header'),
    flt.top_bar(),
    html.Div([
        flt.sidebar(),
        dcc.Loading(html.Main(id='content', className='content'), type='dot',
                    color='#2f6db4', parent_className='content-wrap', delay_show=300),
    ], className='body'),
], className='app')


# ---------------------------------------------------------------------------
# Округ и школа: роль, ручной выбор, клик по плитке округа, сброс
# ---------------------------------------------------------------------------
@app.callback(
    Output('f-district', 'value'), Output('f-district', 'disabled'),
    Output('f-district', 'clearable'),
    Output('f-school', 'value'), Output('f-school', 'options'), Output('f-school', 'clearable'),
    Input('f-role', 'value'), Input('f-district', 'value'), Input('f-school', 'value'),
    Input({'type': 'district-tile', 'index': ALL}, 'n_clicks'), Input('btn-reset', 'n_clicks'),
)
def sync_unit(role, district, school, _tiles, _reset):
    trig = ctx.triggered_id
    if trig == 'btn-reset' or trig == 'f-role':
        district, school = None, None
    elif isinstance(trig, dict):  # клик по плитке округа
        clicks = {i['id']['index']: i.get('value') for i in ctx.inputs_list[3]}
        if not clicks.get(trig['index']) or role == 'director':  # плитка только что появилась
            return (no_update,) * 6
        district, school = trig['index'], None
    elif trig == 'f-district' and school is not None and SCHOOL_DISTRICT.get(school) != district:
        school = None
    elif trig == 'f-school' and school is not None:
        district = SCHOOL_DISTRICT.get(school)

    if role == 'director':
        school = school if school is not None else DEFAULT_SCHOOL
        district = SCHOOL_DISTRICT.get(school)
    elif role == 'district' and not district:
        district = SCHOOL_DISTRICT.get(school) if school is not None else DISTRICTS[0]

    # директору список школ - по всему городу (свою школу выбирает сам, демо без авторизации)
    options = flt.school_options(None if role == 'director' else district)
    return (district, role == 'director', role == 'city',
            school, options, role != 'director')


# ---------------------------------------------------------------------------
# Содержимое вкладки
# ---------------------------------------------------------------------------
def _selection(year, district, school, role, extra_values) -> Selection:
    values = dict(zip(flt.EXTRA_IDS, extra_values))
    df = flt.apply_filters(DF, values, school)
    return Selection(df=df, year=int(year), district=district,
                     school=int(school) if school is not None else None, role=role)


@app.callback(
    Output('content', 'children'),
    Input('tabs', 'value'), Input('f-year', 'value'), Input('f-district', 'value'),
    Input('f-school', 'value'), Input('f-role', 'value'), *EXTRA_INPUTS,
)
def render(tab, year, district, school, role, *extra):
    sel = _selection(year, district, school, role, extra)
    return TABS[tab][1].render(sel)


# ---------------------------------------------------------------------------
# Сброс, счётчик фильтров, выгрузка
# ---------------------------------------------------------------------------
@app.callback(
    Output('f-year', 'value'), *[Output(fid, 'value') for fid in flt.EXTRA_IDS],
    Input('btn-reset', 'n_clicks'), prevent_initial_call=True,
)
def reset_filters(_):
    return YEARS[-1], *flt.EXTRA_DEFAULTS.values()


@app.callback(
    Output('filters-badge', 'children'), Output('active-chips', 'children'), *EXTRA_INPUTS,
)
def show_active(*extra):
    values = dict(zip(flt.EXTRA_IDS, extra))
    chips = []
    if values['f-subject']:
        chips.append('Предмет: ' + ', '.join(map(subject_label, values['f-subject'])))
    if values['f-position']:
        chips.append('Должность: ' + ', '.join(map(subject_label, values['f-position'])))
    if values['f-age'] and list(values['f-age']) != flt.EXTRA_DEFAULTS['f-age']:
        chips.append('Возраст: {}–{}'.format(*values['f-age']))
    if values['f-exp'] and list(values['f-exp']) != flt.EXTRA_DEFAULTS['f-exp']:
        chips.append('Стаж: {}–{}'.format(*values['f-exp']))
    n = flt.active_count(values)
    return (str(n) if n else ''), [html.Span(c, className='chip') for c in chips]


@app.callback(
    Output('download-csv', 'data'), Input('btn-csv', 'n_clicks'),
    State('f-year', 'value'), State('f-district', 'value'), State('f-school', 'value'),
    State('f-role', 'value'), *[State(fid, 'value') for fid in flt.EXTRA_IDS],
    prevent_initial_call=True,
)
def export_csv(_, year, district, school, role, *extra):
    sel = _selection(year, district, school, role, extra)
    my = mt.mobility_year(sel.df, sel.year)
    table = mt.school_summary(sel.unit, Selection(sel.df, my, district, sel.school, role).unit,
                              sel.year, my)
    name = sel.unit_name.replace(' № ', '_').replace(' ', '_')
    return dcc.send_string('\ufeff' + table.to_csv(index=False, sep=';'),
                           f'schools_{name}_{sel.year}.csv')


if __name__ == '__main__':
    app.run(debug=True)
