import os
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder

def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def preprocess_data(input_path, output_path, log_path):
    """
    Предобработка данных:
    - удаление полностью пустых столбцов
    - удаление столбца 'id', если есть
    - кодирование diagnosis
    - нормализация признаков (кроме diagnosis)
    - заполнение пропусков
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    setup_logger(log_path)
    try:
        logging.info("=== Запуск предобработки данных ===")
        df = pd.read_pickle(input_path)
        logging.info(f"Данные успешно загружены из {input_path}")

        # Удаляем полностью пустые столбцы
        empty_cols = [col for col in df.columns if df[col].isnull().sum() == len(df)]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            logging.info(f"Удалены полностью пустые столбцы: {empty_cols}")
        
        # Удаляем столбец id, если есть
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            logging.info("Столбец 'id' удалён.")
        
        # Кодируем diagnosis
        if 'diagnosis' in df.columns:
            le = LabelEncoder()
            df['diagnosis'] = le.fit_transform(df['diagnosis'])
            logging.info("Целевая переменная 'diagnosis' закодирована.")
        
        # Нормализация числовых признаков (кроме diagnosis)
        features = [col for col in df.columns if col != 'diagnosis']
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        logging.info("Числовые признаки нормализованы.")
        
        # Заполнение пропусков (универсально: числовые — среднее, остальные — мода)
        missing = df.isnull().sum().sum()
        if missing > 0:
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
            logging.info(f"Обнаружено и заполнено пропусков: {missing}")
        else:
            logging.info("Пропусков не обнаружено.")
        
        # Проверим результат
        missing_after = df.isnull().sum().sum()
        logging.info(f"Пропусков после заполнения: {missing_after}")
        
        df.to_pickle(output_path)
        logging.info(f"Обработанные данные сохранены в {output_path}")
        logging.info("=== Завершено ===")
        return df
    except Exception as e:
        logging.error(f"Ошибка при предобработке данных: {e}")
        raise
