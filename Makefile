clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage
	rm -rf outputs

clean-logs:
	rm -rf logs/lightning_logs/**

clean-stats:
	rm -rf logs/stats/val_base/**

format:
	pre-commit run -a

test:
	pytest -k "not slow"

test-full:
	pytest

train-baseline:
	python spoof/train.py --device=1 --cfg-training=config/baseline.yaml

fast-train:
	python spoof/train.py --device=1 --cfg-training=config/train.yaml -e=2

train:
	python spoof/train.py --device=1 --cfg-training=config/train.yaml -e=20

eval-small:
	python spoof/validate.py --device=1 --config-data=config/validate.yaml --ckpt=logs/ep018_loss0.00_acc1.000_eer0.000.ckpt

eval-full:
	python spoof/validate.py --device=1 --config-data=config/test.yaml --ckpt=logs/ep018_loss0.00_acc1.000_eer0.000.ckpt
