# ================================================
# Main program to calculate bias-variance tradeoff based on the LASSO
# 
# Author: Kaihang Shi
# last update: 3/10/2021
# =================================================

rm(list=ls())

# ------------- preload  -------------
source("package_verification.R")
source("read_files.R")
source("data_process.R")
source("ml_functions.R")

# Pick the ML model
# machine learning model 
# 1 - Linear Regression (LASSO, RIDGE, MLR)
# 2 - Random Forest
ml_model = 1

# --- directory parameters
# full directory of 'summary.txt'
sum_dir = "../Xe10/summary_1600.txt"
# trained model output directory
data_dir ="../big_data"
# directory where all grid&json files are located
grid_dir = "/home/kaihang/2dhist/Tobacco"
# gcmc file name which contains gcmc data
#gcmcname = 'gcmc_Ethane_298K_40Bar_cm3cm3.txt'
#gcmcname = 'gcmc_Kr_273K_10Bar_cm3cm3.txt'
gcmcname = 'gcmc_Xe_273K_10Bar_cm3cm3.txt'
# set random seed for reproducibility
rand_seed = 5
# number of randomly selected training sets generated
n_trainsets = 300
# training/testing data to split out
n_traindata = 1000
n_testdata = 400
# dimension for energy/gradient
#ener_dim = c('50','26','18','14','10')
#grad_dim = c('29','15','11','8','5')
#ener_dim = c('32','22','17','12','8')
#grad_dim = c('21','11','8','5')
ener_dim = c('22','12','10','8')
grad_dim = c('18','9','5','3')
# prefix for tdhist file
prefix = 'tdhist_Xe_norm_'

# --- Model parameters (Linear Regression)
# alpha value is the elastic net mixing parameter
# (ridge=0, LASSO=1, others=elastic net)
DEFAULT_ALPHA = 1                
  

# --- Model parameters (Random Forest or Extra Tree)
# number of trees to grow
n_tree = 500
# minimum sample size in the leaf node, default = 5 in RF
nsample_leafnode = 5



# ---------------------------- Main Body ------------------------------ 
set.seed(rand_seed)
# Different 2d histogram (model complexity)
vec_name = c()
vec_nfeatures=c()
for (x in ener_dim){
  for (y in grad_dim) {
    vec_name = append(vec_name,paste0(prefix,x,'x',y,'_0.5A.rds'))
    vec_nfeatures = append(vec_nfeatures,as.numeric(x)*as.numeric(y))
  }
}

# initialize variables
mx_mse = matrix(0,nrow = n_trainsets, ncol = length(vec_name))
ary_predy = array(0,dim=c(n_testdata,1,length(vec_name),n_trainsets))

# Read summary.txt which contains information about training and testing set
# read_sum is defined in 'read_files.R'
ls_set = read_sum(sum_dir) 
# extract data frame
vec_trainid = ls_set$train_set$id
vec_testid = ls_set$test_set$id
vec_allid = c(vec_trainid,vec_testid)

# generate unique testing set and the rest of it is the pool for 
vec_testid = sample(vec_allid,n_testdata,replace = FALSE)
vec_remainid = vec_allid[!vec_allid %in% vec_testid]

# prepare training and testing data
gcmc_file = file.path(data_dir,gcmcname)

# set up progress bar
pb = txtProgressBar(min = 1, max = length(mx_mse), style = 3)
pt = 0

# -------------- Loop over all possible training sets
for (iset in 1:n_trainsets) {
  
  # generate random training sets 
  vec_trainid = sample(vec_remainid,n_traindata,replace=FALSE)
  vec_allid = c(vec_trainid,vec_testid)
  imodel = 0
  
  # -------------------- Loop over all models
  for (tdhistname in vec_name) {
    
    imodel = imodel + 1
    
    # function defined in read_files.R
    ls_xy = read_hist_gcmc(grid_dir,
                           tdhistname,
                           gcmc_file,
                           idset=vec_allid)

    df_joint = ls_xy$df_xy
    
    # training data
    df_x_train = dplyr::filter(df_joint, df_joint$id %in% vec_trainid) %>% 
      dplyr::select(.,-c(id,loading))
    vec_y_train = dplyr::filter(df_joint, df_joint$id %in% vec_trainid) %>% 
      .$loading

    # testing data
    df_x_test = dplyr::filter(df_joint, df_joint$id %in% vec_testid) %>% 
      dplyr::select(.,-c(id,loading))
    vec_y_test = dplyr::filter(df_joint, df_joint$id %in% vec_testid) %>% 
      .$loading

    # function defined in 'ml_functions.R' 
    if (ml_model == 1) {
      # LASSO
      ls_out = Linear_Regression(xtrain = df_x_train,
                                 ytrain = vec_y_train,
                                 xtest = df_x_test,
                                 ytest = vec_y_test,
                                 ALPHA = DEFAULT_ALPHA)
      
    } else if (ml_model ==2) {
      
      # standard Random Forest
      ls_out = Random_Forest(xtrain = df_x_train,
                             ytrain = vec_y_train,
                             xtest = df_x_test,
                             ytest = vec_y_test,
                             ntree = n_tree,
                             nodesize = nsample_leafnode)
      
    }
    
    ary_predy[,,imodel,iset] = as.matrix(ls_out$test$pred_y)
    
    # extract rmse
    mse_test  = ls_out$test$eval_metric$mse
    
    mx_mse[iset,imodel] = mse_test
    
    # progress bar
    pt = pt + 1 
    setTxtProgressBar(pb, pt)
    
  }
}

# Calculation of bias squared 
# creat an array containing actual y
ary_actuy = array(vec_y_test,dim=c(n_testdata,1,length(vec_name)))
# averaged over the 4th dimension (different training sets) but keep 1,2,3 dimensions
ary_expec_predy = apply(ary_predy, c(1,2,3),mean)
# calcualte bias squared
ary_bias_sq = (ary_expec_predy - ary_actuy)^2
# take the mean of all testing data,'apply' gives a 1 x n_model matrix
vec_bias_sq = as.vector(apply(ary_bias_sq,c(2,3),mean))

# Calculate variance
ary_expec_predy_expand = array(ary_expec_predy,dim=c(n_testdata,1,length(vec_name),n_trainsets))
ary_var_D = (ary_expec_predy_expand - ary_predy)^2
ary_expec_var = apply(ary_var_D,c(1,2,3),mean)
vec_var = as.vector(apply(ary_expec_var,c(2,3),mean))


# calculate the expected value over all training sets
vec_mse_alltrain = colMeans(mx_mse)
#df = data.frame(nfeatures= vec_nfeatures, mse = vec_mse_alltrain)
# print all results
print('========================== Below are the results ======================')
print(paste0('Random seed is: ',rand_seed))
print('-----------------------------------------------------------------------')

print('Original Matrix for MSE with different training sets:')
mx_mse

print('-----------------------------------------------------------------------')
print('Final expected MSE of different energy/gradient bin widths:')
mse = matrix(vec_mse_alltrain,
             nrow = length(ener_dim),
             ncol = length(grad_dim),
             byrow = TRUE)
mse

print('-----------------------------------------------------------------------')
print('Corresponding bias squared:')
bias = matrix(vec_bias_sq,
              nrow = length(ener_dim),
              ncol = length(grad_dim),
              byrow = TRUE)
bias

print('-----------------------------------------------------------------------')
print('Corresponding variance:')
variance = matrix(vec_var,
                  nrow = length(ener_dim),
                  ncol = length(grad_dim),
                  byrow = TRUE)
variance












