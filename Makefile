.PHONY: install start_webserver start_scheduler

install:
	pip install -r requirements.txt

start_webserver:
	export AIRFLOW_HOME=$(PWD)/airflow_home && airflow webserver --port 8080

start_scheduler:
	export AIRFLOW_HOME=$(PWD)/airflow_home && airflow scheduler
