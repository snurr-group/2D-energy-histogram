## This folder contains essential R scripts for converting energy/energy gradient grids into a 2D energy histogram, and for training and testing ML models:

### main_prep_2dhist.R       
Main program to convert grids into 2D energy histogram in RDS format

### main_ml_2dhist.R          
Main program for reading 2D energy histograms in RDS format and converting them into feature matrix; and for performing ML training

### main_model_test.R        
Main program for testing pre-trained machine learning models

### main_bias_var_tradeoff.R 
Main program for performing bias-variance tradeoff calculations to determine bin width for 2D energy histogram.

### main_diagnose.R          
Main program for analyzing energy and energy histogram distribution

### analyze_functions.R    
Collections of functions for analysis (e.g., distribution)

### data_process.R            
Collections of functions for data processing 

### make_plot.R              
Collections of functions for making plots

### ml_functions.R        
Collections of functions for ML workflow

### nn_hypertunning.R        
Function for initializting multilayer perceptron (MLP) tuning

### read_files.R              
Collections of functions for reading data

### package_verification.R   
Essential R packages needs to be loaded
