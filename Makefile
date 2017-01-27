all: debug.log

debug.log: examples/akerlind.py fleet/__init__.py
	PYTHONPATH=. python3 examples/akerlind.py >debug.log

test:
	py.test-3 .

.PHONY: test
