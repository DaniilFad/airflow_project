import os
import pandas as pd
import logging
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json

def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def calc_and_save_metrics(model_path, X_test_path, y_test_path, metrics_path, log_path):
    """
    Вычислить метрики (Accuracy, Precision, Recall, F1), сохранить их в JSON, вывести отчёт в лог.
    """
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    setup_logger(log_path)
    try:
        logging.info("=== Запуск расчёта метрик ===")
        # Загрузка модели и данных
        model = joblib.load(model_path)
        X_test = pd.read_pickle(X_test_path)
        y_test = pd.read_pickle(y_test_path)
        logging.info(f"Модель и тестовые данные успешно загружены.")

        # Предсказания
        y_pred = model.predict(X_test)

        # Метрики
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logging.info(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        # Классический classification_report — для логов и контроля
        logging.info("\n" + classification_report(y_test, y_pred))

        # Сохраняем метрики в JSON
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Метрики сохранены в {metrics_path}")
        logging.info("=== Завершено ===")
        return metrics
    except Exception as e:
        logging.error(f"Ошибка при расчёте метрик: {e}")
        raise
