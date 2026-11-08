"""Загрузка готовой витрины и применение сохранённой модели без подготовки признаков."""
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd

# Данные лежат в корневой папке data/ (на уровень выше dashboard/)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'models'


def load_data():
    path = DATA_DIR / 'teachers.parquet'
    df = pd.read_parquet(path)
    return df


def load_predictions(df, model_dir=MODEL_DIR):
    """Выбрать готовые столбцы по метаданным и получить прогноз будущего года.

    Ни обучения, ни заполнения пропусков, ни преобразования признаков здесь нет.
    При отсутствии или ошибке модели аналитика продолжает работать без прогноза.
    """
    result = df.copy()
    result['prob_leave'] = np.nan
    directory = Path(model_dir)
    if not (directory / 'catboost.cbm').exists() and not (directory / 'metadata.json').exists():
        return result, 'Прогноз недоступен: сохранённая бинарная модель ещё не подключена.'
    try:
        from catboost import CatBoostClassifier
        meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        if (meta['schema_version'] != 1 or meta['task'] != 'binary_leave'
                or meta['target'] != 'target' or meta['positive_class'] != 1):
            raise ValueError('Ожидается бинарная модель с target=1 для ухода')
        features, cats = meta['features'], meta['categorical_features']
        forbidden = {'id', 'year', 'target', 'is_out_next_year', 'mobility_status',
                     'movement_type', 'next_school', 'school_num_next',
                     'prob_leave', 'prob_move', 'prob_stay', 'prob_remain', 'risk_level'}
        if (not features or len(features) != len(set(features))
                or forbidden.intersection(features) or not set(cats) <= set(features)):
            raise ValueError('Некорректный список признаков')
        year, years = meta['forecast_year'], meta['training_years']
        if type(year) is not int or not years or not all(type(y) is int and y < year for y in years):
            raise ValueError('Годы обучения должны предшествовать году прогноза')
        if df.duplicated(['id', 'year']).any():
            raise ValueError('Нужен уникальный ключ id, year')
        model = CatBoostClassifier()
        model.load_model(str(directory / 'catboost.cbm'), format='cbm')
        classes = np.asarray(model.classes_).tolist()
        if len(classes) != 2 or set(classes) != {0, 1} or classes != meta['classes']:
            raise ValueError('Классы модели должны совпадать с метаданными: 0 и 1')
        if list(model.feature_names_) != features:
            raise ValueError('Порядок признаков не совпадает с моделью')
        if {features[i] for i in model.get_cat_feature_indices()} != set(cats):
            raise ValueError('Категориальные признаки не совпадают с моделью')
        mask = df['year'].eq(year)
        if not mask.any():
            return result, 'Прогноз недоступен: в витрине нет года применения модели.'
        if 'target' in df and df.loc[mask, 'target'].notna().any():
            raise ValueError('Для года применения уже известны исходы')
        X = df.loc[mask, features]  # Только выбор столбцов; все значения уже готовы в parquet.
        if X[cats].isna().any().any():
            raise ValueError('Нужно исправить категориальные пропуски в исходной витрине')
        proba = np.asarray(model.predict_proba(X), dtype=float)
        if (proba.shape != (len(X), 2) or not np.isfinite(proba).all()
                or (proba < 0).any() or (proba > 1).any()
                or not np.allclose(proba.sum(axis=1), 1, atol=1e-7)):
            raise ValueError('Некорректные вероятности модели')
        result.loc[mask, 'prob_leave'] = proba[:, classes.index(1)]
        return result, None
    except Exception:
        logging.getLogger(__name__).exception('Не удалось загрузить или применить бинарную модель')
        return df.assign(prob_leave=np.nan), 'Прогноз недоступен: модель не удалось применить к текущим данным.'

# Глобальные справочники для фильтров
DF, MODEL_STATUS = load_predictions(load_data())
YEARS = sorted(int(y) for y in DF['year'].unique())
SCHOOLS = sorted(int(s) for s in DF['school_num'].unique())
DISTRICTS = sorted(DF['district'].dropna().unique())
POSITIONS = sorted(DF['main_position_cat'].unique())
SUBJECTS = sorted(s for s in DF['speciality'].dropna().unique() if s != 'не указано')
