# Concordance Coefficient Simulation & Visualization

This repository accompanies the thesis paper for the 2025 CSE3000 Research Project course. It contains code used to compute and analyse various concordance coefficients on ranked retrieval results.

## Overview

- **`main.py`**: Computes $\tau$, $\tau_{ap}$, and $\tau_h$ correlation coefficients between pairs of ranked lists. These are extended to relevance profiles, as outlined in the paper.
- **`graphs.ipynb`**: Reproduces the plots presented in the thesis paper using the computed coefficients.
- **`script.sh`**: Shell script to run the simulations on the DelftBlue HPC cluster. 
- **`untied_data/`**: Locally contains TREC datasets for the 2010 - 2014 ad hoc task in the web track, albeit with tied items (same score) having been randomly broken using random.shuffle using seed = 42. Also contains data from the simulation algorithms, which do not contain any ties.
- **`output/`**: Automatically generated directory containing the result CSVs per dataset.

---

## How to Use

### 1. Compute Concordance Coefficients

To process a folder of untied ranking data:

```bash
python main.py
```

 - This will iterate over all folders listed in the folders variable (e.g., simulated_data, 2010, etc.).

 - It computes 12 variations of Kendall-style concordance coefficients for each pairwise comparison of ranking files in a topic-wise fashion.

 - The results are written as CSV files to the output/ directory.

You can upload your own untied data into a subfolder under untied_data/. The expected CSV format per file is:
```bash
topic,doc_id,score,rel
```
### 2. Visualize Results
Open the Jupyter notebook:

```bash
jupyter notebook graphs.ipynb
```

This notebook:

 - Reads the processed coefficient CSVs.

 - Creates aggregated transparency scatter plots for both relevance-based and item-based coefficients.

 - Provides two final tables summarising absolute differences between the base and relevance-based metrics (for TREC and simulated datasets).

### 3. Run on DelftBlue Cluster
The **`bash.sh`** file can be used to run the computation on the Delftblue HPC cluster. This is useful for handling large-scale datasets that are computationally intensive to process.

---

## Requirements
To run this project, you’ll need:

 - Python 3.7+

 - Jupyter Notebook

 - The following Python packages:

```bash
pip install numpy matplotlib pandas
```

Multiprocessing is used internally by main.py to parallelise computation.
