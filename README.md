# mESC FACS Data Processing

This repository contains a series of scripts used to process FACS data originally collected and reported on by Sáez et al. in "[Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions](https://pubmed.ncbi.nlm.nih.gov/34536382/)."
The data generated in these scripts is subsequently used in "[Dynamical systems theory informed learning of cellular differentiation landscapes](https://www.biorxiv.org/content/10.1101/2024.09.21.614191v1)."
A separate Github repository corresponding to this work can be found at [https://github.com/AddisonHowe/dynamical-landscape-inference](https://github.com/AddisonHowe/dynamical-landscape-inference).


## Setup

```bash
conda create -p ./env python=3.12 numpy=1.26 matplotlib=3.8 pandas=2.2 scikit-learn=1.4 scipy=1.12 seaborn=0.13 tqdm=4.66 umap-learn=0.5 ipykernel ipywidgets
conda activate env
pip install flowutils==1.1
```

## About

The following table summarizes the contained scripts. 

| Script | Description |
| --- | --- |
| [script1_preprocessing.py](script1_preprocessing.py) | Loads raw FACS data and compiles and saves dataframes. |
| [script1b_signal_plots.py](script1b_signal_plots.py) | Generates plots of the signal dynamics corresponding to each experiment. |
| [script2_clustering.py](script2_clustering.py) | Recapitulates the original clustering algorithm detailed by Sáez et al., assigning a type to each observed cell. |
| [script2a_postcluster_plots.py](script2a_postcluster_plots.py) | Generates plots after the celltype labeling process. |
| [script3a_isolate1.py](script3a_isolate1.py) | Isolates cells involved in the first binary decision. |
| [script3b_isolate2.py](script3b_isolate2.py) | Isolates cells involved in the second binary decision.  |
| [script4a_dimred_pca.py](script4a_dimred_pca.py) | Performs dimension reduction on the FACS data, using PCA. |


## Usage

```bash
# Preprocess data
python script1_preprocessing.py
python script1b_signal_plots.py

# Perform clustering
python script2_clustering.py
python script2a_postcluster_plots.py

# Isolate first and second cellular decisions
python script3a_isolate1.py
python script3b_isolate2.py

# Dimension reduction for first cellular decision
python script4a_dimred_pca.py -k facs_v3 -d 1 --fit_on_subset

# Dimension reduction for second cellular decision
python script4a_dimred_pca.py -k facs_v4 -d 2 --fit_on_subset
```


# Acknowledgments
This work was inspired by the work of Sáez et al. in [Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions](https://pubmed.ncbi.nlm.nih.gov/34536382/).
The FACS data collected by the original authors was provided upon request.


# References
[1] Sáez M, Blassberg R, Camacho-Aguilar E, Siggia ED, Rand DA, Briscoe J. Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions. Cell Syst. 2022 Jan 19;13(1):12-28.e3. doi: 10.1016/j.cels.2021.08.013. Epub 2021 Sep 17. PMID: 34536382; PMCID: PMC8785827.
