# 2D Energy Histogram
R implementation of machine learning using 2D energy histogram features for prediction of adsorption. If you have any questions, please feel free to contact me at kaihangshi0@gmail.com!<br/>

![Workflow to construct 2D energy histogram features](https://github.com/snurr-group/2D-energy-histogram/blob/main/feature_engineering_scheme.jpg)

## Files
This repository contains essential code to reproduce the results in the paper. Due to the size limit of GitHub repository, all supporting data, including CIF files and textural properties for amorphous porous materials, trained ML models, and GCMC adsorption data are stored at Zenodo space. You can access those data at XXX. CIF files and textural properties for MOFs used in this work can be downloaded at https://mof.tech.northwestern.edu/databases.

## Usage
You can reproduce the plots in the paper by using the code provided. For example, let's draw a parity plot comparing ML predictions with GCMC ground truth. You can download a trained ML model in RDS format from XXX. We pick a file "LASSO_Kr1bar_17x3_2dhist.rds"; this RDS file contains a trained LASSO model using 2D energy histogram features (2dhist) for Kr adsorption at 1 bar, 273 K (Kr1bar); the 2D energy histogram in this case is a 17x3 matrix. To make the plot, simply typing the following code in RStudio,
```
> source('make_plot.R')
> plot_parity('LASSO_Kr1bar_17x3_2dhist.rds')
```
The RDS file contains all training & testing data; you can access them by typing
```
> source('package_verification.R')
> data = read_rds('LASSO_Kr1bar_17x3_2dhist.rds')
> data$train$actu_y  # print GCMC training data
```
We have also prepared an Excel file that contains all GCMC adsorption data (available at Zenodo and SI of the paper). For multilayer perceptron (MLP) model, the actual model was saved in H5 format, all other data were still saved in RDS format.

## Reference
If you find the code and data are useful to your project, please cite this paper: 
Shi, Li, Anstine, Tang, Colina, Sholl, Siepmann, Snurr, "Two-dimensional Energy Histograms as Features for Machine Learning to Predict Adsorption in Diverse Nanoporous Materials", to be published.
