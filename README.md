:construction::construction::construction: **We are currently refactoring and cleaning the codebase. It will be ready in the next months. Please feel free to contact us directly if you have specific requests.** :construction::construction::construction:

---

# Learning Spectral Unions of Partial Deformable 3D Shapes

<p align="center">
    <a href="https://github.com/lucmos/spectral-unions/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/lucmos/spectral-unions/Test%20Suite/main?label=main%20checks></a>
    <a href="https://lucmos.github.io/spectral-unions"><img alt="Docs" src=https://img.shields.io/github/deployments/lucmos/spectral-unions/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.2.1-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

In this paper, we introduce a learning-based method to estimate the Laplacian spectrum of the union of partial non-rigid 3D shapes, without actually computing the 3D geometry of the union.


## Installation

```bash
pip install git+ssh://git@github.com/lucmos/spectral-unions.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git@github.com:lucmos/spectral-unions.git
cd spectral-unions
conda env create -f env.yaml
conda activate spectral-unions
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
