clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage
	rm -rf outputs

format:
	pre-commit run -a

eval:
	python src/spoof/validate.py --ckpt=logs/ep004_loss0.01_acc1.000_eer0.000.ckpt --device=3 --config-data=config/test.yaml
