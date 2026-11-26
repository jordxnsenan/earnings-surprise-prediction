# Can Historical Earnings Patterns Predict Future Earnings Surprises?

I built this project to examine whether a company's past earnings results can help predict its next earnings surprise. My main question was whether looking at four preceding recorded earnings events gives a classifier more useful information than looking at the most recent event alone.

I classify an earnings announcement as a **beat** when reported earnings per share (EPS) exceeds the recorded analyst estimate. If actual EPS is equal to or below the estimate, I classify it as **meet-or-miss**.

## Dataset

I used the public [Sov.ai Earnings Surprise dataset](https://huggingface.co/datasets/sovai/earnings_surprise). It contains repeated snapshots of earnings information, so I first had to reconstruct individual earnings events. I then built prediction examples using each company's four preceding recorded events.

| Statistic                                     |           Value |
| --------------------------------------------- | --------------: |
| Original source snapshots                     |       1,657,661 |
| Candidate earnings events                     |         111,989 |
| Eligible prediction samples                   |          90,349 |
| Tickers represented in the prediction samples |           5,064 |
| Original input features                       |              25 |
| Numerical and discrete features               |              23 |
| Categorical features                          |               2 |
| Earnings beats                                | 55,576 (61.51%) |
| Meets or misses                               | 34,773 (38.49%) |

I kept the evaluation chronological. The 82,353 observations from before 2024 were used for development, and I reserved all 7,996 observations from 2024 for the final evaluation. More details about the source and its preparation are in [docs/dataset.md](docs/dataset.md).

## Approach

I started with exploratory data analysis to understand the feature distributions, missing values, correlations, and unusually large earnings values. I examined preprocessing choices and used PCA to visualize the data. The EDA notebook and [EDA report](reports/EDA_Report.pdf) document that work.

For clustering, I tried K-Means, agglomerative clustering, and DBSCAN. I compared their internal evaluation metrics and inspected the assignments in two-dimensional PCA plots. I also ran Isolation Forest, Local Outlier Factor, and One-Class SVM to identify unusual observations. Since an unusual financial result can still be meaningful, I tested whether removing outliers from the training data helped XGBoost. It produced slightly worse development results, so I kept those observations.

For classification, I compared logistic regression, a decision tree, random forest and XGBoost against simple historical baselines. The main evaluation used expanding chronological validation folds for 2020 through 2023. I reported macro-F1 as the primary metric along with accuracy, class-level precision and recall, and ROC-AUC. I also compared confusion matrices and ROC curves.

I used mutual information to select features within each training fold then compared the models with and without selection. The experiments include fitting time and memory measurements. I ran randomized hyperparameter searches using validation years 2020–2022 and assessed the chosen configurations on 2023 data. Finally, I tested how predictions changed when I kept only the latest earnings event or removed the explicit trend features.

## Results

I selected the original XGBoost configuration with the **without-trends** feature set for the final holdout. It uses 22 of the original historical inputs excluding `eps_change1_2`, `eps_trend4` and `surprise_trend4`. I trained it on the pre-2024 development data and evaluated it once on the reserved 2024 observations.

| Model                     |   Macro-F1 |   Accuracy |    ROC-AUC |
| ------------------------- | ---------: | ---------: | ---------: |
| XGBoost without trends    | **0.6074** | **64.32%** | **0.6812** |
| Previous outcome baseline |     0.6014 |     61.71% |     0.6009 |
| Majority class baseline   |     0.3710 |     58.99% |     0.5000 |

XGBoost had the highest macro-F1 and accuracy of these three models although its macro-F1 advantage over the previous-outcome baseline was only about 0.006. The previous-outcome baseline identified more meet-or-miss cases: its recall for that class was 51.08%, compared with 41.63% for XGBoost. That tradeoff matters when interpreting the overall scores.

The ablation experiments gave me a more direct answer to the research question. Using the full four-event history performed better across the development folds than using only the latest event. Removing the three explicit trend features barely changed the average results, which is why I chose the smaller feature set for the final evaluation.

The final metrics and saved predictions are in [results/final_holdout/](results/final_holdout/). The other experiment results are under [`results/`](results/).

## Limitations

One clear limitation that I identified with respect to the source data is that it's snapshot dates do not reliably show when every earnings value became publicly available, and some snapshots contain actual EPS before the listed publication date. I therefore decided to treat this as a retrospective classification study. The results do not establish how the models would perform on information available in real time or as part of a trading strategy.

## Repository layout

```text
earnings-surprise-prediction/
├── docs/                  # Dataset and model-selection notes
├── notebooks/             # Analysis notebooks with saved outputs
├── reports/               # Written course reports
├── results/               # Metrics, comparisons and figures
├── scripts/               # Utility scripts
├── src/earnings_surprise/  # Main Python package
│   ├── clustering/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── outliers/
│   └── tuning/
├── tests/                 # Automated tests
└── pyproject.toml         # Dependencies and package settings
```

Raw and processed datasets are stored under `data/` on the local machine and are excluded from Git. The scripts below recreate those files.

## Installation

The project supports Python 3.11 and 3.12. I used Python 3.12.16 on macOS 27.2.1 (arm64) during development.

I included `constraints.txt` to record the dependency versions used to reproduce my experiments. I recommend installing the project with these constraints when reproducing the reported results.

Clone the current working branch and enter the project directory:

```bash
git clone --branch repository-organization https://github.com/jordxnsenan/earnings-surprise-prediction.git

cd earnings-surprise-prediction
```

Create an environment, install the package, and run the tests:

```bash
python3.12 -m venv .venv

source .venv/bin/activate

ppython -m pip install -c constraints.txt -e ".[dev]"

python -m pytest -q
```

Run the module commands below from the **repository root**. They use paths relative to that directory.

### Reproducibility Verification

I verified the reference environment by installing the project in a fresh virtual environment using `constraints.txt`. All 243 automated tests passed.

I then rebuilt the datasets from the pinned raw source in a separate directory. The reconstructed earnings events, source row mapping, prediction windows and historical feature matrix matched my original processed datasets exactly.

After retraining the selected XGBoost configuration and evaluating both baselines I reproduced the committed final holdout metrics and all 7,996 predicted labels for each model. The prediction probabilities also matched exactly. I performed these checks on macOS using an ARM64 processor.

### Prepare the dataset

The download script retrieves a pinned revision of the Sov.ai dataset and checks its size and SHA-256 hash. I then inspect the raw records, audit the snapshots, reconstruct earnings events and build the feature matrix.

```bash
python -m earnings_surprise.data.download

python -m earnings_surprise.data.inspect

python -m earnings_surprise.data.audit_snapshots

python -m earnings_surprise.data.events

python -m earnings_surprise.data.samples

python -m earnings_surprise.features

python -m earnings_surprise.evaluation.splits
```

Processed Parquet files are written to `data/processed/`. The corresponding summaries go to `results/data/`.

## Reproducing the experiments

I committed the result files and figures used in the notebooks, so the analyses can be read without rerunning every experiment. To regenerate them, install the package and prepare the dataset first. The commands within each section are listed in their required order.

Some runs take considerably longer than others especially the model benchmarks and tuning searches. You can run one section at a time once its inputs are available.

### Clustering

These commands reproduce the clustering runs, parameter sweeps, PCA figures, and comparisons:

```bash
python -m earnings_surprise.clustering.algorithms.kmeans
python -m earnings_surprise.clustering.experiments.kmeans_sweep
python -m earnings_surprise.clustering.analysis.visualize

python -m earnings_surprise.clustering.algorithms.agglomerative
python -m earnings_surprise.clustering.experiments.agglomerative_sweep
python -m earnings_surprise.clustering.analysis.compare_partitions

python -m earnings_surprise.clustering.experiments.dbscan_diagnostics
python -m earnings_surprise.clustering.experiments.dbscan_sweep
python -m earnings_surprise.clustering.analysis.dbscan_visualize
python -m earnings_surprise.clustering.analysis.compare_algorithms
```

The results are under `results/clustering/` and the plots are in `results/figures/clustering/`.

### Outlier detection

I first ran each detector, then compared the assignments and visualized where the methods agreed. The last two commands test the effect of removing Isolation Forest outliers from the training data.

```bash
python -m earnings_surprise.outliers.algorithms.isolation_forest
python -m earnings_surprise.outliers.analysis.analyze_isolation
python -m earnings_surprise.outliers.algorithms.local_outlier_factor
python -m earnings_surprise.outliers.algorithms.one_class_svm
python -m earnings_surprise.outliers.analysis.compare_detectors
python -m earnings_surprise.outliers.analysis.compare_all_detectors
python -m earnings_surprise.outliers.analysis.visualize_agreement

python -m earnings_surprise.outliers.analysis.evaluate_filter_impact
python -m earnings_surprise.outliers.analysis.plot_filter_impact
```

The result files are in `results/outliers/` and the plots are in `results/figures/outliers/`.

### Classification and feature selection

Run the baselines and classifiers before generating the model comparison. The later commands produce the ROC plots and feature-selection efficiency measurements.

```bash
python -m earnings_surprise.evaluation.classification.run_baselines
python -m earnings_surprise.evaluation.classification.run_logistic
python -m earnings_surprise.evaluation.classification.run_decision_tree
python -m earnings_surprise.evaluation.classification.run_random_forest
python -m earnings_surprise.evaluation.classification.run_xgboost

python -m earnings_surprise.evaluation.analysis.compare_models
python -m earnings_surprise.evaluation.analysis.analyze_errors
python -m earnings_surprise.evaluation.analysis.plot_comparison

python -m earnings_surprise.evaluation.analysis.roc_comparison
python -m earnings_surprise.evaluation.benchmarking.benchmark_xgboost
python -m earnings_surprise.evaluation.benchmarking.benchmark_other_classifiers --model logistic_regression
python -m earnings_surprise.evaluation.benchmarking.benchmark_other_classifiers --model decision_tree
python -m earnings_surprise.evaluation.benchmarking.benchmark_other_classifiers --model random_forest
python -m earnings_surprise.evaluation.analysis.compare_feature_efficiency
python -m earnings_surprise.evaluation.analysis.roc_other_classifiers
```

Saved results are in `results/classification/` and the figures are in `results/figures/classification/`.

### Hyperparameter tuning

I used expanding chronological folds for the searches and then checked the selected settings on 2023 development observations. The final command produces the tuning comparison.

```bash
python -m earnings_surprise.tuning.splits
python -m earnings_surprise.tuning.xgboost_search
python -m earnings_surprise.tuning.search_other_classifiers --model logistic_regression
python -m earnings_surprise.tuning.search_other_classifiers --model decision_tree
python -m earnings_surprise.tuning.search_other_classifiers --model random_forest

python -m earnings_surprise.tuning.xgboost_assessment
python -m earnings_surprise.tuning.assess_other_classifiers --model logistic_regression
python -m earnings_surprise.tuning.assess_other_classifiers --model decision_tree
python -m earnings_surprise.tuning.assess_other_classifiers --model random_forest
python -m earnings_surprise.tuning.compare_tuning_results
```

The saved search and assessment results are under `results/tuning/`. The comparison figure is in `results/figures/tuning/`.

### Feature ablations and final holdout

I compared different amounts of earnings history before making the final model choice:

```bash
python -m earnings_surprise.evaluation.classification.run_ablations
```

The ablation results are saved to `results/ablations/xgboost_feature_ablations.json`.

The final command retrains the selected, fixed configuration on the pre-2024 observations and evaluates it on the reserved 2024 data:

```bash
python -m earnings_surprise.evaluation.classification.final_holdout
```

It writes the evaluation metrics and predictions to `results/final_holdout/`. I kept the holdout separate from feature selection and tuning. Its results should not be used to change the model settings or classification threshold.

Regenerating an experiment can change tracked output files. Check `git status` and review any differences before committing them.

## Notebooks

The notebooks contain the analysis and saved outputs. After generating the required result files you can run them in VS Code or JupyterLab using the project's Python environment.

| Notebook                         | What it covers                                  |
| -------------------------------- | ----------------------------------------------- |
| `data_preprocessing_eda.ipynb`   | Source inspection, preprocessing, and EDA       |
| `clustering.ipynb`               | K-Means, agglomerative clustering, and DBSCAN   |
| `outlier_detection.ipynb`        | Outlier methods, comparisons, and filtering     |
| `classification.ipynb`           | Baselines, classifiers, and feature selection   |
| `hyperparameter_tuning.ipynb`    | Search results and model assessments            |
| `feature_ablations.ipynb`        | How different historical feature sets performed |
| `final_holdout_evaluation.ipynb` | The reserved 2024 evaluation                    |

The EDA notebook can download and prepare the source data. The other notebooks mainly read saved metrics and figures generated by the Python modules. Run notebooks from the repository root or the `notebooks/` directory so their project-root checks can find the files.

## Google Colab

I previously tested the EDA submission in Colab. The full reorganized workflow still needs a fresh Colab run before I can claim it has been verified there.

In a new Colab session, check the Python version first. The project requires Python 3.11 or 3.12.

```python
import sys

print(sys.version)
```

With a compatible runtime, clone the working branch and install the package:

```python
!git clone --branch repository-organization https://github.com/jordxnsenan/earnings-surprise-prediction.git
%cd /content/earnings-surprise-prediction
%pip install -e ".[dev]"
```

If Colab asks for a runtime restart after installation, restart it and return to `/content/earnings-surprise-prediction`.

Commands from the sections above can be run in Colab by placing `!` in front of each shell command. For example, to prepare the data:

```python
!python -m earnings_surprise.data.download
!python -m earnings_surprise.data.inspect
!python -m earnings_surprise.data.audit_snapshots
!python -m earnings_surprise.data.events
!python -m earnings_surprise.data.samples
!python -m earnings_surprise.features
!python -m earnings_surprise.evaluation.splits
```

The `.ipynb` files can be opened through Colab's GitHub notebook browser. They include saved outputs if you only need to review the completed analyses. When executing cells, keep the working directory at the project root and use the environment where the package is installed.

## Author

Jordan Archer