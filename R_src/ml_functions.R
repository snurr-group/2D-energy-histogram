# ===========================================
# Functions for machine learning purposes
# ===========================================

source("package_verification.R")
source("data_process.R")


# ================ Train linear regression function using glmnet package ===================
fit_glmnet = function(df_x, vec_y, lambda = NULL, alpha = 1, fit_intercept = TRUE) {

  
  # ======================================================================
  # By default, only removes zero variance columns without mean-centering to zero or unit variance (norm).
  #
  # Output: return a list consisting of model details
  # Input:
  # df_x    - data frame containing stacked vectors of trainning MOF info
  # vec_y   - target y value in vector format
  # lambda  - penalty parameter (coefficient) in regularization term
  # alpha   - elasticnet mixing parameter (ridge=0, LASSO=1, others=elastic net)
  # fit_intercept - flag to turn on fitting intercept
  # ======================================================================
  
  vec_orig_y = vec_y
  
  # If lambda is not defined, let's do cross-validation
  cvfit = NULL
  if (is.null(lambda)) {
    # Idea from https://www.r-bloggers.com/ridge-regression-and-the-lasso/
    # It would be good to run cv.glmnet many times to get the best lambda
    trial_lambdas = 10^seq(5, -3, length = 100)  
    cvfit = cv.glmnet(as.matrix(df_x), 
                      vec_y, 
                      alpha=alpha, 
                      nfolds=10, 
                      type.measure="mse", 
                      lambda=trial_lambdas, 
                      intercept=fit_intercept)
    # This is also where you could alternatively set lambda.1se, depending on which resource
    # see https://cran.r-project.org/web/packages/glmnet/glmnet.pdf, page 15, for more info
    lambda = cvfit$lambda.min  
  }
  
  # Actually train the model
  # glmnet will standardize the data set by default
  modfit = glmnet(as.matrix(df_x),
                  vec_y, 
                  alpha=alpha, 
                  lambda=lambda, 
                  intercept=fit_intercept)
  
  # evaluate model on the training set
  vec_pred_y = predict(modfit,as.matrix(df_x)) %>% as.numeric
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric = eval_metric(vec_orig_y, vec_pred_y)
  
  # Return a list of the relevant model details
  list(mod = modfit,
       x = df_x,
       actu_y = vec_orig_y,
       pred_y = vec_pred_y,
       lambda = lambda,
       alpha = alpha,
       cv = cvfit,
       eval_metric = ls_metric)
}



# =================== Model evaluation on the testing data set =================
pred_glmnet = function(train_model, df_x, vec_y){
  
  # ======================================================================
  # Predict targets using fitted model
  #
  # Output: return a list consisting of predictions
  # Input:
  # train_model - list object from fit_glmnet function
  # df_x    - data frame containing stacked vectors of testing MOF info
  # vec_y   - target y value in vector format
  # ======================================================================
  
  # evaluate model on the testing set
  vec_pred_y = predict(train_model$mod,as.matrix(df_x)) %>% as.numeric
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric = eval_metric(vec_y, vec_pred_y)
  
  # Return a list of the relevant model details
  list(x = df_x,
       actu_y = vec_y,
       pred_y = vec_pred_y,
       eval_metric = ls_metric)
  
}





# =================== Linear regression wrapper =================
# Combine as a big function so no need to replicate all for learning curve studies
Linear_Regression = function(xtrain,ytrain,xtest,ytest,ALPHA){
  
  # ======================================================================
  # Wrapper for linear regression function
  #
  # Output: return a list 
  # Input: see descriptions in consisting functions
  # ======================================================================
  
  # fit_glmnet function does the jobs including standardizing the data, finding
  #   the lambda using cross-validation and fitting the model
  # functions defined in 'ml_functions.R'
  ls_mod = fit_glmnet(df_x = xtrain, 
                      vec_y = ytrain, 
                      lambda = NULL, 
                      alpha = ALPHA, 
                      fit_intercept = TRUE)
  
  # Evaluate the trained model on testing data set 
  ls_test = pred_glmnet(train_model = ls_mod,
                        df_x = xtest,
                        vec_y = ytest)
  
  # output a list
  list(train = ls_mod,
       test  = ls_test)
}





# ================== Random Forest Wrapper ====================
Random_Forest = function(xtrain,ytrain,xtest,ytest,ntree,nodesize){
  
  # ======================================================================
  # Wrapper for Random Forest algorithm
  #
  # Output: return a list 
  # Input: see descriptions in consisting functions
  # ======================================================================
  
  # Train the model with mtry tuning
  # RF does not need standardize the data set
  # Decrease in MSE
  rf_mod = randomForest::tuneRF(x = xtrain,
                                y = ytrain,
                                doBest = TRUE,
                                ntree = ntree,
                                replace = TRUE,
                                nodesize = nodesize,
                                importance = TRUE)
  
  # evaluate model on the training set (OOB error)
  vec_y_pred = predict(rf_mod) %>% as.numeric
  # vec_y_pred = predict(rf_mod,as.matrix(df_x_train)) %>% as.numeric
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric = eval_metric(ytrain, vec_y_pred)
  
  # return a list for the trained model
  ls_mod = list(mod = rf_mod,
                x = xtrain,
                actu_y = ytrain,
                pred_y = vec_y_pred,
                eval_metric = ls_metric)
  
  # Evaluate the testing data
  vec_y_pred = predict(rf_mod,as.matrix(xtest)) %>% as.numeric
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric = eval_metric(ytest, vec_y_pred)
  
  # return a list for the trained model
  ls_test = list(x = xtest,
                 actu_y = ytest,
                 pred_y = vec_y_pred,
                 eval_metric = ls_metric)
  
  # output a list
  list(train = ls_mod,
       test  = ls_test)
}





# ================== Extra Trees Wrapper ====================
Extra_Trees = function(xtrain,ytrain,xtest,ytest,ntree,nodesize,nfeature,tunerandomcut = FALSE){
  
  # ======================================================================
  # Wrapper for Extra Tree algorithm
  #
  # Output: return a list 
  # Input: see descriptions in consisting functions
  #   nfeature  - number of features
  #   tunerandomcut - flag to turn on the tuning of random cut
  # ======================================================================
  
  # set up data frame for parameter tuning
  if (tunerandomcut) {
    et_grid = expand.grid(mtry = seq(10,nfeature,10),numRandomCuts = 1:5)
  } else {
    et_grid = expand.grid(mtry = seq(10,nfeature,10),numRandomCuts = 1)
  }
  
  # set up 5-fold cross-validation
  cv_5 = trainControl(method = "cv", number = 5)
  
  # Train the model with mtry/randomcut tuning
  et_mod = caret::train(x = xtrain,
                        y = ytrain,
                        method = "extraTrees",
                        ntree = ntree,
                        nodesize = nodesize,
                        numThreads = 4,
                        trControl = cv_5,
                        tuneGrid = et_grid)
  
  # evaluate model on the training set 
  vec_y_pred = predict(et_mod,as.matrix(xtrain)) %>% as.numeric
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric = eval_metric(ytrain, vec_y_pred)
  
  # return a list for the trained model
  ls_mod = list(mod = et_mod,
                x = xtrain,
                actu_y = ytrain,
                pred_y = vec_y_pred,
                eval_metric = ls_metric)
  
  # Evaluate the testing data
  vec_y_pred = predict(et_mod,as.matrix(xtest)) %>% as.numeric
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric = eval_metric(ytest, vec_y_pred)
  
  # return a list for the trained model
  ls_test = list(x = xtest,
                 actu_y = ytest,
                 pred_y = vec_y_pred,
                 eval_metric = ls_metric)
  
  # output a list
  list(train = ls_mod,
       test  = ls_test)
}





# ================== Gradient boosting Wrapper ====================
XGradient_Boosting = function(xtrain,ytrain,xtest,ytest,xgbGrid){
  
  # ======================================================================
  # Wrapper for gradient boosting tree algorithm
  #
  # Output: return a list 
  # Input: 
  # xtrain, xtest - data frame for input
  # ytrain, ytest - vector of labels
  # xgbGrid - data frame for possible hyperparameter search space
  # ======================================================================
  
  # specify cross-validation
  cv_5 = trainControl(method = "cv", number = 5, allowParallel = TRUE)
  
  # tranin xgboost with gbtree (gblinear resembles LASSO)
  xgb_mod = caret::train(x = as.matrix(xtrain),
                         y = ytrain,
                         method = "xgbTree",
                         objective = "reg:squarederror",
                         tuneGrid = xgbGrid,
                         trControl = cv_5)
  
  # evaluate model on the training set
  vec_y_pred = predict(xgb_mod,as.matrix(xtrain)) %>% as.numeric
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric = eval_metric(ytrain, vec_y_pred)
  
  # return a list for the trained model
  ls_mod = list(mod = xgb_mod,
                x = xtrain,
                actu_y = ytrain,
                pred_y = vec_y_pred,
                eval_metric = ls_metric)
  
  # Evaluate the testing data
  vec_y_pred = predict(xgb_mod,as.matrix(xtest)) %>% as.numeric
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric = eval_metric(ytest, vec_y_pred)
  
  # return a list for the trained model
  ls_test = list(x = xtest,
                 actu_y = ytest,
                 pred_y = vec_y_pred,
                 eval_metric = ls_metric)
  
  # output a list
  list(train = ls_mod,
       test  = ls_test)
}























