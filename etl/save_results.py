import os
import shutil
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

def save_results(src_model, src_metrics, dst_folder, log_path):
    """
    Копирует артефакты модели и метрик в results, если нужно.
    Не копирует, если файл уже на месте.
    """
    os.makedirs(dst_folder, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    setup_logger(log_path)
    try:
        logging.info("=== Финальная сборка артефактов ===")
        model_dst = os.path.join(dst_folder, os.path.basename(src_model))
        metrics_dst = os.path.join(dst_folder, os.path.basename(src_metrics))

        copied = False
        if os.path.abspath(src_model) != os.path.abspath(model_dst):
            shutil.copy(src_model, model_dst)
            logging.info(f"Модель скопирована в {model_dst}")
            copied = True
        else:
            logging.info(f"Модель уже лежит в {model_dst}, копирование не требуется.")

        if os.path.abspath(src_metrics) != os.path.abspath(metrics_dst):
            shutil.copy(src_metrics, metrics_dst)
            logging.info(f"Метрики скопированы в {metrics_dst}")
            copied = True
        else:
            logging.info(f"Метрики уже лежат в {metrics_dst}, копирование не требуется.")

        if not copied:
            print("Артефакты уже в папке results. Копирование не требуется.")
        else:
            print("Артефакты успешно скопированы в папку results.")
        logging.info("=== Завершено ===")
    except Exception as e:
        logging.error(f"Ошибка при сохранении артефактов: {e}")
        raise
