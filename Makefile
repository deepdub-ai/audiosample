
all:
	python3 -m build .
	python3 -m twine upload --repository pypi dist/* --verbose
test:
	pytest -m "not slow" .
testslow:
	pytest -m slow
clean:
	rm -rf audiosample.egg-info/
	rm -rf dist/
