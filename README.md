# 2D Energy Histogram
R implementation of machine learning using 2D energy histogram features for prediction of adsorption.<br/>

![Workflow to construct 2D energy histogram features](https://github.com/snurr-group/2D-energy-histogram/blob/main/feature_engineering_scheme.jpg)


## Files
This repository contains R and Python code to reproduce the results in the paper. Due to the size limit of GitHub repository, all supporting data, including CIF files and textural properties for amorphous porous materials, trained ML models, and GCMC adsorption data are stored at Zenodo space. You can acess those data at XXX. CIF files and textural properties for MOFs used in this work can be downloaded at https://mof.tech.northwestern.edu/databases.

## Usage
You can reproduce the plots in the paper by simply using the code provided. For example, let's draw a parity plot comparing ML predictions with GCMC ground truth. You can download a trained ML model in RDS format from XXX. We pick file "LASSO_Kr1bar_17x3_2dhist.rds"; this RDS file contains a trained LASSO model using 2D energy histogram features (2dhist) for Kr adsorption at 1 bar, 273 K (Kr1bar); the 2D energy histogram in this case is a 17x3 matrix. To make plots, simply typing the following code in RStudio,
```
>source("make_plot.R")
>plot_parity('LASSO_Kr1bar_17x3_2dhist.rds')
```



## Reference

