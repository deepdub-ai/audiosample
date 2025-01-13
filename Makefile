
all:
	python3 -m build .
	python3 -m twine upload --repository pypi dist/* --verbose
test:
	pytest -m "not slow" .
test-stream:
	pytest -m "not slow" tests/test_stream.py
testslow:
	pytest -m slow
clean:
	rm -rf audiosample.egg-info/
	rm -rf dist/
