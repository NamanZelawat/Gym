run-pytorch:
	cd pytorch && uv run --active jupyter lab --no-browser

run-project:
	cd project && uv run --active jupyter lab --no-browser