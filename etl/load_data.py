import os
import pandas as pd
import logging

def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_data(input_path, output_path, log_path):
    """
    Загружает данные из CSV, делает базовый анализ и сохраняет в .pkl.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    setup_logger(log_path)
    try:
        logging.info("=== Запуск скрипта загрузки данных ===")
        df = pd.read_csv(input_path)
        logging.info(f"Данные успешно загружены из {input_path}")
        logging.info(f"Размер: {df.shape}")
        logging.info(f"Пропуски по столбцам:\n{df.isnull().sum()}")
        logging.info(f"Типы данных:\n{df.dtypes}")
        logging.info(f"Первые 3 строки:\n{df.head(3)}")
        df.to_pickle(output_path)
        logging.info(f"Промежуточный файл сохранён в {output_path}")
        logging.info("=== Завершено ===")
        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        raise
