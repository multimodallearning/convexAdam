# Steps to set up testing environment

Set up conda environment:
```
conda create --name=convex_adam python=3.10
```

Activate environment:
```
conda activate convex_adam
```

Install module and dependencies:
```
pip install -e .
pip install -r requirements_dev.txt
```

Perform tests:
```
pytest
mypy src
flake8 src
```

# Push release to PyPI
1. Increase version in setup.py, and set below
2. Build: `python -m build`
3. Test package distribution: `python -m twine upload --repository testpypi dist/*0.2.0*`
4. Distribute package to PyPI: `python -m twine upload dist/*0.2.0*`
