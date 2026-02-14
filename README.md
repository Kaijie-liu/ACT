# the framework

An end-to-end neural network verification platform that supports refinement-based precision, diverse models, input formats, and specification types.

## Quick Start

## 0. Preparation
Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) and create running environment.

```
conda env create -f environment.yml    # Install required lib packages to run the framework
conda activate cuc-py312 # Activate an environment (python-3.12) 
```

## 1. Clone repository
```
git clone --recursive <ANONYMOUS_REPO_URL>
cd framework
```

## 2. Apply and download the [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) (Optional for MILP optimization)
```
cp /path/to/your/gurobi.lic ./modules/gurobi/gurobi.lic  # put gurobi.lic file in ./modules/gurobi/ directory
```

## 3. Run the framework phases
```
python -m cuc.pipeline --help
```

## 4. Small Jupyter notebook demos
- [Fuzzer example](ipynb/vnnlib_fuzzer.ipynb)
- [Verifier example](ipynb/vnnlib_verifier.ipynb)
- [More](ipynb/)

### License

### Acknowledgements
This project was developed with the assistance of GitHub Copilot to enhance code readability and efficiency. AI-generated suggestions were reviewed and tested by the contributors before inclusion.
