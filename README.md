# BG_CERP

Code repository for analyses used in the **Basal Ganglia CERP study**.

This repository contains scripts used to generate figures and statistical analyses for the manuscript.

## Repository Structure

- `boxplot_pub.py`  
  Generates boxplot visualizations of feature distributions.

- `ratio_pub.py`  
  Computes and plots feature ratios used in the analysis.

- `rf_pub.py`  
  Random forest–based feature analysis.

- `rf_roc_pub_pvalue.py`  
  ROC analysis with statistical significance testing.

- `tsne_atlas.py`  
  t-SNE visualization of atlas or feature space.

## Requirements

Tested with:

- Python ≥ 3.9
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy

