import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

BASE_DIR = Path(__file__).resolve().parent
LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино',
}


@st.cache_resource
def load_model(path, version):
    return joblib.load(path)


@st.cache_data
def load_test(path, version):
    return pd.read_csv(path)


def positive_proba(model, X):
    column = np.flatnonzero(np.asarray(model.classes_) == 1).item()
    return model.predict_proba(X)[:, column]


@st.cache_data
def test_proba(_model, X, model_version):
    return positive_proba(_model, X)


def validate_features(frame, features):
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('CSV не содержит строк с данными.')
    try:
        X = frame.loc[:, features].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Все признаки должны быть числами. Проверьте разделитель CSV.') from exc
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения.')
    if not X['is_red'].isin([0, 1]).all():
        raise ValueError('Столбец is_red должен содержать только 0 (белое) или 1 (красное).')
    return X


def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title='Винная подборка', page_icon='🍷', layout='wide')
    st.title('Винная подборка')
    st.caption('Хорошее вино — экспертная оценка не ниже 7. Решение: вероятность ≥ порог.')
    paths = {name: BASE_DIR / name for name in ('model.joblib', 'config.json', 'test.csv')}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        st.error('Рядом с app.py не хватает файлов: ' + ', '.join(missing)
                 + '. Скопируйте их из папки wine_app, сохранённой в ноутбуке.')
        st.stop()
    try:
        model_version = paths['model.joblib'].stat().st_mtime_ns
        model = load_model(str(paths['model.joblib']), model_version)
        df = load_test(str(paths['test.csv']), paths['test.csv'].stat().st_mtime_ns)
        threshold_default = float(json.loads(paths['config.json'].read_text(encoding='utf-8'))['threshold'])
        if not np.isfinite(threshold_default) or not 0 <= threshold_default <= 1:
            raise ValueError('Порог в config.json должен быть от 0 до 1.')
        features = list(model.feature_names_in_)
        if set(features) != set(LABELS):
            raise ValueError('Модель должна принимать 12 признаков Wine Quality.')
        X_test = validate_features(df, features)
        if 'target' not in df or set(df['target'].unique()) != {0, 1}:
            raise ValueError('test.csv должен содержать target с обоими классами: 0 и 1.')
        y_test = df['target'].astype(int)
        scores = test_proba(model, X_test, model_version)
    except Exception as exc:
        st.error(f'Не удалось загрузить модель или данные: {exc}')
        st.stop()

    threshold = st.sidebar.slider(
        'Порог отбора', 0.0, 1.0, value=threshold_default,
        step=0.001, format='%.6f', key='threshold',
    )
    st.sidebar.caption(f'Порог из ноутбука: {threshold_default:.6f}. Общий для всех вкладок.')
    st.sidebar.caption('Изменение порога в демо не заменяет его выбор на валидации.')
    single, quality, batch, coefficients = st.tabs([
        'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель',
    ])

    with single:
        st.subheader('Оцените одно вино')
        st.caption('Диапазоны и начальные значения взяты из test.csv: минимум, максимум и медиана.')
        values = {}
        columns = st.columns(3)
        for i, feature in enumerate(features):
            with columns[i % 3]:
                if feature == 'is_red':
                    color = st.radio('Цвет вина', ['Белое', 'Красное'],
                                     index=int(X_test[feature].median() >= 0.5), horizontal=True)
                    values[feature] = int(color == 'Красное')
                else:
                    lo, hi = float(X_test[feature].min()), float(X_test[feature].max())
                    if lo == hi:
                        values[feature] = lo
                        st.write(f'{LABELS[feature]}: {lo}')
                    else:
                        values[feature] = st.number_input(
                            LABELS[feature], min_value=lo, max_value=hi,
                            value=float(X_test[feature].median()),
                            step=max((hi - lo) / 1000, 0.000001),
                            format='%.6f', key=f'input_{feature}',
                        )
        one = pd.DataFrame([values], columns=features)
        probability = float(test_proba(model, one, model_version)[0])
        st.metric('Вероятность хорошего вина по модели', f'{probability:.1%}')
        if probability >= threshold:
            st.success('Рекомендовать: модель относит вино к хорошим.')
        else:
            st.info('Не включать в подборку при текущем пороге.')

    with quality:
        st.subheader('Как порог меняет подборку')
        st.write('Повышение порога уменьшает подборку: recall не растёт, '
                 'а precision обычно повышается, но это не гарантировано на каждом шаге. '
                 'Снижение порога позволяет найти больше хороших вин, '
                 'но в подборку могут попасть дополнительные посредственные вина.')
        pred = (scores >= threshold).astype(int)
        precision = metrics.precision_score(y_test, pred, zero_division=0)
        recall = metrics.recall_score(y_test, pred, zero_division=0)
        f1 = metrics.f1_score(y_test, pred, zero_division=0)
        cols = st.columns(4)
        for col, label, value in zip(cols[:3], ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
            col.metric(label, f'{value:.3f}')
        cols[3].metric('Вин в подборке', f'{pred.sum()} / {len(pred)}')
        if pred.sum() == 0:
            st.info('Ни одно вино не отобрано. Precision и F1 отображаются как 0.')
        cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fig, ax = plt.subplots(figsize=(5, 3))
        metrics.ConfusionMatrixDisplay(cm, display_labels=['Не хорошее', 'Хорошее']).plot(
            ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_xlabel('Предсказанный класс')
        ax.set_ylabel('Истинный класс')
        show_figure(fig)
        st.caption(f'FP = {fp}: посредственные вина в подборке. FN = {fn}: пропущенные хорошие вина.')
        fpr, tpr, _ = metrics.roc_curve(y_test, scores)
        pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, scores)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, scores):.4f}')
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[0].scatter(fp / (fp + tn), recall, color='crimson', zorder=5, label='Текущий порог')
        axes[0].set(xlabel='False Positive Rate', ylabel='Recall', title='ROC-кривая')
        axes[1].plot(pr_recall, pr_precision,
                     label=f'AP = {metrics.average_precision_score(y_test, scores):.4f}')
        axes[1].axhline(y_test.mean(), linestyle='--', color='gray', label='Случайный уровень')
        axes[1].scatter(recall, precision, color='crimson', zorder=5, label='Текущий порог')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
        for ax in axes:
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.legend()
            ax.grid(alpha=0.2)
        fig.tight_layout()
        show_figure(fig)
        st.caption('При пустой подборке красная PR-точка показана с precision=0 по соглашению; '
                   'конечная точка самой PR-кривой у sklearn имеет precision=1. '
                   'Кривые и их площади не меняются при движении порога.')

    with batch:
        st.subheader('Оцените список вин')
        st.write('Загрузите CSV с 12 признаками без target. Разделитель — запятая.')
        st.download_button('Скачать пример CSV', X_test.head(5).to_csv(index=False).encode('utf-8'),
                           file_name='wine_example.csv', mime='text/csv')
        uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
        if uploaded is not None:
            try:
                incoming = pd.read_csv(uploaded)
                X_batch = validate_features(incoming, features)
                result = incoming.copy()
                result['probability_good'] = test_proba(model, X_batch, model_version)
                result['prediction'] = (result['probability_good'] >= threshold).astype(int)
                result['threshold'] = threshold
                st.dataframe(result, use_container_width=True)
                st.download_button('Скачать прогнозы', result.to_csv(index=False).encode('utf-8-sig'),
                                   file_name='wine_predictions.csv', mime='text/csv')
            except Exception as exc:
                st.error(f'Не удалось обработать CSV: {exc}')

    with coefficients:
        st.subheader('Какие признаки связаны с хорошим вином')
        estimator = model.named_steps['logisticregression']
        weights = pd.Series(estimator.coef_[0], index=features).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([LABELS[name] for name in weights.index], weights.values,
                color=np.where(weights.values >= 0, '#287D65', '#B65353'))
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Коэффициент после StandardScaler')
        fig.tight_layout()
        show_figure(fig)
        positive = weights[weights > 0].nlargest(3)
        negative = weights[weights < 0].nsmallest(3)
        st.write('Сильнее всего за хорошее вино: ' + (', '.join(LABELS[x] for x in positive.index) or 'нет'))
        st.write('Сильнее всего против: ' + (', '.join(LABELS[x] for x in negative.index) or 'нет'))
        st.caption('Коэффициент описывает изменение логарифма шансов при увеличении признака '
                   'на одно стандартное отклонение и неизменных остальных признаках. '
                   'Это связь в модели, а не доказательство причинности. '
                   'Для бинарного is_red переход 0 → 1 не равен одному стандартному отклонению.')


if __name__ == '__main__':
    main()
