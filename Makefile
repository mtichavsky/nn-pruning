.PHONY: venv

venv:
	python -m venv venv
	venv/bin/pip install -r requirements.txt

train:
	venv/bin/python main.py train --epochs "$NOF_EPOCHS"