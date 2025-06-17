import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def train_model(input_path, model_path, X_test_path, y_test_path, log_path, test_size=0.2, random_state=42):
    """
    Обучение модели LogisticRegression, сохранение модели и тестовой выборки.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(X_test_path), exist_ok=True)
    os.makedirs(os.path.dirname(y_test_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    setup_logger(log_path)
    try:
        logging.info("=== Запуск обучения модели ===")
        df = pd.read_pickle(input_path)
        logging.info(f"Данные успешно загружены из {input_path}")

        # Заполним пропуски, если вдруг остались
        if df.isnull().sum().sum() > 0:
            logging.warning("Обнаружены пропуски! Заполняем средними значениями.")
            df = df.fillna(df.mean())

        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(f"Тренировочная выборка: {X_train.shape}, Тестовая: {X_test.shape}")

        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_train, y_train)
        logging.info("Модель LogisticRegression обучена.")

        # Сохраняем модель
        joblib.dump(model, model_path)
        logging.info(f"Модель сохранена в {model_path}")

        # Сохраняем тестовые данные отдельно
        X_test.to_pickle(X_test_path)
        y_test.to_pickle(y_test_path)
        logging.info(f"Тестовые данные сохранены: {X_test_path}, {y_test_path}")

        logging.info("=== Завершено ===")
        return model, X_test, y_test
    except Exception as e:
        logging.error(f"Ошибка при обучении модели: {e}")
        raise
