# BioPrimeLASSO

BioPrimeLASSO combines the original R implementation of biologically primed LASSO modelling with a new Python package that mirrors the same scientific workflow. This repository now contains:

1. The peer-reviewed R package and scripts that reproduce the published analyses.
2. A Python port exposing a composable, object-oriented interface with scikit-learn compatible components, persistence helpers, and extensible Manhattan plotters.
3. Documentation, tests, and example notebooks demonstrating how to tune, fit, and visualise bio-primed models on modern datasets.

---

## Table of Contents
1. [Conceptual Background](#conceptual-background)
2. [Repository Layout](#repository-layout)
3. [Python Package Architecture](#python-package-architecture)
4. [Installation](#installation)
5. [End-to-End Workflow](#end-to-end-workflow)
6. [Plotting Backends](#plotting-backends)
7. [Data Interoperability](#data-interoperability)
8. [Testing](#testing)
9. [R Package Walkthrough](#r-package-walkthrough)
10. [Reproducibility Resources](#reproducibility-resources)

---

## Conceptual Background

Precision oncology frequently relies on regularised regression to prioritise biomarkers from thousands of genomic features. BioPrimeLASSO augments vanilla LASSO by integrating prior biological knowledge—such as STRING protein–protein interaction scores—into the penalty weights so that features connected to a gene of interest are preferentially selected. The Python package follows the same logic as the R code path:

1. Estimate an optimal LASSO penalty (`lambda`) using standard cross-validation.
2. Transform prior interaction scores into rescaled penalty multipliers.
3. Sweep candidate `phi` values to blend data-driven penalties with priors and compute fold-wise RMSE.
4. Choose the best-performing `phi`, refit the model, and store both baseline and bio-primed coefficients.
5. Produce Manhattan-style plots that juxtapose baseline and bio-primed hits for downstream interpretation.

---

## Repository Layout

```
BioPrimeLASSO/
├── AGENTS.md                     # Agent log and collaboration notes
├── README.md                     # This document
├── R/                            # Original R source code
├── Reproducibility/              # Manuscript-oriented analysis scripts
├── src/bioprimelasso/            # Python package modules
├── tests/                        # pytest-based unit and integration tests
├── notebooks/                    # Usage examples and walkthrough notebooks
├── pyproject.toml                # Python packaging metadata
└── ...                           # Images, licensing, and auxiliary files
```

---

## Python Package Architecture

The Python implementation is organised into composable modules so each responsibility can be tested independently or swapped out for custom alternatives:

| Module | Responsibility |
| --- | --- |
| `bioprimelasso.model` | High-level façade (`BioPrimeLassoModel`) coordinating tuning, fitting, persistence, prediction, and plotting. |
| `bioprimelasso.tuning` | Hyperparameter utilities that find `lambda` and identify the best `phi` via cross-validated RMSE matrices. |
| `bioprimelasso.backend` | Thin wrapper around scikit-learn’s `Lasso` to perform scaled fits and predictions with custom penalty weights. |
| `bioprimelasso.scores` | Interfaces for transforming biological interaction networks into penalty multipliers (`NetworkScoreProvider`). |
| `bioprimelasso.repository` | Persistence layer that stores and retrieves structured results (`ResultRepository`, `StoredResult`). |
| `bioprimelasso.plotting` | Strategy objects (starting with a Matplotlib implementation) that render Manhattan plots from stored results. |

Key design features:
- **Dependency Injection:** Constructors accept protocol-style components so advanced users can bring their own tuner, backend, repository, or plotters.
- **Typed Data Contracts:** Methods are annotated with `pandas`, `numpy`, and `pathlib` types to clarify expectations and ease IDE support.
- **Result Handles:** Every call to `fit` returns a handle that resolves stored coefficients, metrics, and metadata for later prediction or visualisation.
- **Extensible Plotting:** A registry of `ManhattanPlotter` implementations allows static Matplotlib output today and interactive Plotly/Bokeh variants tomorrow.

Refer to [`src/bioprimelasso/model.py`](src/bioprimelasso/model.py) for concrete method signatures and docstrings.

---

## Installation

### Python environment

1. Create and activate a virtual environment (conda, venv, or poetry).
2. Install the package in editable mode along with optional extras for notebook exploration.

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install --upgrade pip
pip install -e .[dev]
```

The `dev` extra pulls in `pytest`, `seaborn`, and `jupyter` for testing and examples.

### R environment

If you plan to use the R scripts alongside the Python port, follow the original installation instructions in the [R Package Walkthrough](#r-package-walkthrough).

---

## End-to-End Workflow

The notebook in [`notebooks/bioprimelasso_demo.ipynb`](notebooks/bioprimelasso_demo.ipynb) provides a runnable example. The typical sequence in Python is:

1. **Prepare data** – supply a predictor matrix `X` (`pandas.DataFrame`), response vector `y` (`pandas.Series`), and a STRING-style interaction network.
2. **Instantiate components** – choose or customise tuner, backend, repository location, score provider, and desired plotters.
3. **Fit model** – call `BioPrimeLassoModel.fit(...)` to perform lambda selection, phi tuning, and final fitting.
4. **Inspect results** – access stored coefficients, RMSE matrices, and metadata via the returned handle and repository helpers.
5. **Predict** – run `model.predict(X_new, handle=handle)` to score new samples.
6. **Cross-validate** – use `model.cross_validate(...)` to obtain diagnostic matrices for research workflows.
7. **Visualise** – invoke `model.plot_manhattan(handle, plotter="matplotlib", output="...pdf")` to create figures similar to the R package.

Inline docstrings and type hints describe expected arguments for each method, and the test suite under `tests/` showcases smaller unit-level examples.

---

## Plotting Backends

The `bioprimelasso.plotting` package currently exposes:

- `MatplotlibManhattanPlotter`: produces publication-ready scatter plots with seaborn styling, chromosome colouring, and optional label highlights. Outputs to PDF/PNG paths supplied via the `output` parameter.

To add a new backend (e.g., Plotly), implement the `ManhattanPlotter` protocol and register it when constructing `BioPrimeLassoModel`:

```python
from bioprimelasso.plotting import PlotRegistry

plotters = PlotRegistry.default()
plotters["plotly"] = PlotlyManhattanPlotter(...)
model = BioPrimeLassoModel(..., plotters=plotters)
```

---

## Data Interoperability

The R workflow persisted results as `.RData` files. The Python repository provides flexible alternatives:

- **Pickle (`.pkl`)** for rapid prototyping of `StoredResult` objects.
- **Parquet (`.parquet`)** for columnar storage of coefficient and correlation tables.
- **JSON/CSV** for metadata or when integrating with web dashboards.

`ResultRepository` abstracts away the file layout; the default implementation serialises to compressed JSON alongside parquet tables so plotting backends can hydrate pandas DataFrames efficiently. Adaptors can be added to load legacy `.RData` artifacts if needed.

---

## Testing

Unit tests are implemented with `pytest` and cover tuning utilities, backend scaling, repository persistence, score normalisation, plotting hooks, and integration flows.

```bash
pytest
```

The suite fabricates synthetic datasets, ensuring deterministic behaviour by fixing NumPy random seeds. Continuous integration should execute this command to guard against regressions.

---

## R Package Walkthrough

The original R package remains available for analysts who prefer the R ecosystem. Key steps:

1. Install dependencies (`glmnet`, `ggplot2`, `ggrepel`).
2. Load provided toy datasets (CNV, dependency scores, STRING-derived interactions).
3. Generate prior scores via `get_scores()`.
4. Run `bplasso()` to perform bio-primed fitting and save results.
5. Visualise with `plot_manhattan()`.

Detailed code snippets are preserved below for convenience.

<details>
<summary>Show R walkthrough</summary>

```r
install.packages("devtools")
devtools::install_github("dmhenke/BioPrimeLASSO")

cnv <- read.csv("./cnv_EGFR.tsv", sep = '\t', header = TRUE)
ppi <- read.csv("./ppi_w_symbols_EGFR.tsv", sep = '\t', header = TRUE)
demeter2 <- read.csv("./demeter2_EGFR.tsv", sep = '\t', header = TRUE)

mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
gene_info <- getBM(
  attributes = c("chromosome_name", "start_position", "hgnc_symbol"),
  filters = "hgnc_symbol",
  values = colnames(cnv),
  mart = mart
)
```

```
# Additional R walkthrough content unchanged from the original README...
```

</details>

---

## Reproducibility Resources

The `Reproducibility/` folder contains the precise scripts used in the published study. Consult the included README for instructions on downloading data, configuring environments, and rerunning the manuscript figures.

---

For any questions or contributions, please open an issue or submit a pull request referencing the relevant Python modules or R scripts. Happy modelling!
