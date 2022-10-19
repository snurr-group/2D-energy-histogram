# 2D Energy Histogram
R implementation of machine learning using 2D energy histogram features for prediction of adsorption. If you have any questions, please feel free to contact me at kaihangshi0@gmail.com!<br/>

![Workflow to construct 2D energy histogram features](https://github.com/snurr-group/2D-energy-histogram/blob/main/feature_engineering_scheme.jpg)

## Files
This repository contains essential code and example data to reproduce the results in the paper. The supplementary data including CIF files and textural properties for amorphous porous materials, trained ML models, and GCMC adsorption data that were used for ML training and testing are stored on Zenodo (https://doi.org/10.5281/zenodo.5481697). CIF files and textural properties for MOFs used in this work can be directly downloaded from MOFX-DB: https://mof.tech.northwestern.edu/databases.

## Usage
### Generation of 2D energy histograms
To generate 2D energy histograms, first we complie the C++ code under the "Energy_energyGradients_generator" folder. This C++ binary program can generate 3D energy and energy gradient grids. With grids ready, we can use "main_prep_2dhist.R" under "R_src" folder to generate 2D energy histograms.

### Reproduction of the results
You can reproduce the plots in the paper by using the code provided. For example, we can download trained ML models from Zenodo (https://doi.org/10.5281/zenodo.5481697) and select a trained ML model: "LASSO_Kr1bar_17x3_2dhist.rds". This RDS file contains a trained LASSO model using 2D energy histogram features (2dhist) for Kr adsorption at 1 bar, 273 K (Kr1bar); the 2D energy histogram in this case is a 17x3 matrix. To make the plot, simply typing the following commands in RStudio (make sure the RDS file is under the same directory as the R source code),
```
> source('make_plot.R')
> plot_parity('LASSO_Kr1bar_17x3_2dhist.rds')
```
The RDS file also contains all training & testing data; you can access them by typing
```
> source('package_verification.R')
> data = read_rds('LASSO_Kr1bar_17x3_2dhist.rds')
> data$train$actu_y  # print GCMC training data
> data$trainid       # print ToBaCCo MOF id (tobmof-id) in the training set
> data$test$pred_y   # print ML predictions on the testing data set
```
For multilayer perceptron (MLP) model, the actual model was saved in H5 format, all other data were still saved in RDS format. We have also prepared an Excel file that contains all GCMC adsorption data. This file is available with the published paper and can also be downloaded from the Zenodo repo. 

### Application of trained ML models to new data sets
To use the trained ML model on new data sets, R code "main_model_test.R" under "R_src" folder can do the job.

## Reference
If you find the code and data are useful to your project, please cite this paper: 
Shi, Li, Anstine, Tang, Colina, Sholl, Siepmann, Snurr, "Two-dimensional Energy Histograms as Features for Machine Learning to Predict Adsorption in Diverse Nanoporous Materials", to be published.
