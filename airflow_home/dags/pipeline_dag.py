from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Добавим путь к etl-модулям, чтобы можно было их импортировать из папки etl/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'etl'))

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='ml_etl_pipeline',
    default_args=default_args,
    description='ML ETL Pipeline for Breast Cancer Diagnosis',
    schedule_interval=None,
    catchup=False
) as dag:

    def task_load_data():
        from load_data import load_data
        input_path = "data/data.csv"
        output_path = "data/loaded.pkl"
        log_path = "logs/load_data.log"
        load_data(input_path, output_path, log_path)

    def task_preprocess():
        from preprocess import preprocess_data
        input_path = "data/loaded.pkl"
        output_path = "data/preprocessed.pkl"
        log_path = "logs/preprocess_data.log"
        preprocess_data(input_path, output_path, log_path)

    def task_train_model():
        from train_model import train_model
        input_path = "data/preprocessed.pkl"
        model_path = "results/model.pkl"
        X_test_path = "data/X_test.pkl"
        y_test_path = "data/y_test.pkl"
        log_path = "logs/train_model.log"
        train_model(input_path, model_path, X_test_path, y_test_path, log_path)

    def task_calc_metrics():
        from calc_metrics import calc_and_save_metrics
        model_path = "results/model.pkl"
        X_test_path = "data/X_test.pkl"
        y_test_path = "data/y_test.pkl"
        metrics_path = "results/metrics.json"
        log_path = "logs/calc_metrics.log"
        calc_and_save_metrics(model_path, X_test_path, y_test_path, metrics_path, log_path)

    def task_save_results():
        from save_results import save_results
        src_model = "results/model.pkl"
        src_metrics = "results/metrics.json"
        dst_folder = "results"
        log_path = "logs/save_results.log"
        save_results(src_model, src_metrics, dst_folder, log_path)

    t1 = PythonOperator(
        task_id='load_data',
        python_callable=task_load_data
    )
    t2 = PythonOperator(
        task_id='preprocess_data',
        python_callable=task_preprocess
    )
    t3 = PythonOperator(
        task_id='train_model',
        python_callable=task_train_model
    )
    t4 = PythonOperator(
        task_id='calc_metrics',
        python_callable=task_calc_metrics
    )
    t5 = PythonOperator(
        task_id='save_results',
        python_callable=task_save_results
    )

    t1 >> t2 >> t3 >> t4 >> t5
