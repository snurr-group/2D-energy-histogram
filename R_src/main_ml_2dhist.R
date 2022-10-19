# ================================================
# Main program to train/test machine learning model
# 
# Author: Kaihang Shi
# =================================================

rm(list=ls())

# =========================== preload  ===============================
source("package_verification.R")
source("read_files.R")
source("data_process.R")
source("ml_functions.R")
#source("make_plot.R")

# ============================ Input parameters =============================
# Pick the machine learning model
# 'lr'    - Linear Regression (LASSO, RIDGE, MLR)
# 'rf'    - Random Forest
# 'cnn'   - Convoluted Neural Nets (CNN)
# 'mlp'   - Multilayer perceptron 
# 'et_R'  - Extra Tree (A variation of RF), R version 
# 'et_sk' - Extra Tree, sklearn, Python version
# 'xgb'   - Gradient boosting tree method
# 'gs'    - TODO: Generalized Stacking/transfer learning (see algorithm for details)
# 'umap'  - Uniform Manifold Approximation and Projection (UMAP)
ml_model = 'mlp'

# --------------- Global Parameters -----------------
# Operational system ('win' or 'linux')
os_current = 'linux'
# setting directory based on the operational system
if (os_current == 'win') {
  # directory to adsorbate specific folder (location of 'summary.txt', 'persimg.csv' etc)
  ads_dir = "E:/Research_data/2Dhistogram/kr1/"
  # gcmc data (and other related data) directory
  data_dir ="../All_data/"
  # directory where all grid & json files are located
  grid_dir =  "E:/Research_data/2Dhistogram/tobdown" 
  
} else if (os_current == 'linux') {
  ads_dir = "../eth4/"
  data_dir ="../big_data"
  grid_dir = "/home/kaihang/2dhist/Tobacco"
}

# gcmc file name which contains gcmc data
gcmcname = 'gcmc_Ethane_298K_4Bar_cm3cm3.txt'
# file name for processed 2d histogram 
tdhistname = "tdhist_CH3_norm_14x5_0.5A.rds"  
# output name for R-style save file
output = 'test'  # 'UMAP_hex0.02_extrafeatures'

# Feature selection (major category)
# Options: '2dhist' (2D energy/energy gradient histogram) 
#          'persimg_2d' (persistent image of 2D homology)
#          'persimg_1d' (persistent image of 1D homology)
#          'textural'  (baseline textural features, including void fraction, pld, lcd etc.)
main_features = c('textural')
# textural features
# Options: 'vf', 'lcd', 'pld', 'sa_tot_m2cm3', 'sa_tot_m2g', 'topo'
tex_features = c('vf','lcd','pld','sa_tot_m2cm3','sa_tot_m2g')
# Method to scale ONLY 'persimg_2d' and 'struc' features 
# options: 'standardize', 'normalize', 'none'
scaler = 'normalize'

# Cutoff for feature pruning (to make model stable, set both zero for 'umap' method)
# remove feature column with mean <= cutoff_mean and variance <= cutoff_var
# Only applied to persimg_2d & persimg_1d now (4/1/2021)
meancut = 0
stdcut = 0

# set random seed for reproducibility 
random_seed = 5

# Plot learning curve? 
learning_curve = FALSE
# number of training data as a start
n_train_start = 600
# interval for learning
n_intv = 50

# ------------ Model Specific Parameters ------------
# ------- Linear Regression -------
# alpha value is the elastic net mixing parameter; ridge=0, LASSO=1, others=elastic net
DEFAULT_ALPHA = 1                
# if take log for y data
#iflogy = FALSE  
# flag to exlude zero gcmc loading
# (this opition is to exlude wrong zero data in MOFdb for the moment)
#ifnonzero_load = FALSE       

# ------- Random Forest or Extra Tree -------
# number of trees to grow
n_tree = 500
# minimum sample size in the leaf node, default = 5 in RF
nsample_leafnode = 5
# if perform parameter tuning for number of random cuts (for extra trees, R version only)
iftunerandomcut = FALSE
# number of total features for mtry tuning (for extra tree, R version only)
n_feature = 110

# ------- Gradient Boosting tree -------
# search grid for optimization of hyperparameters
xgbGrid = expand.grid(nrounds = c(50,100,150,200,300),
                      max_depth = c(5, 10, 15, 20),
                      colsample_bytree = seq(0.2, 1, length.out = 10),
                      eta = c(0.1,0.15,0.2,0.25,0.3,0.35),
                      gamma=0,
                      min_child_weight = 5,
                      subsample = 1)

# xgbGrid = expand.grid(nrounds = c(600),  
#                       max_depth = c(10),
#                       colsample_bytree = c(0.5),
#                       eta = c(0.05),
#                       gamma=0,
#                       min_child_weight = 10,
#                       subsample = 1)

# ------- Neural Networks (CNN/MLP) -------
# if tunning the hyperparameters
tunehyperpara = FALSE

# ------- UMAP -------
# which version to use ('R' or 'Py')
umap_version = 'Py' 
umap_neighbors = 30 # default: 15, typically (5,50)
umap_epochs = 500   # default: 200 for large dataset, 500 for small
umap_min_dist = 0.2  # default: 0.1
umap_target_weight = 0 # 0: only feature; 1: only target learning in supervised mode; 0.5: balanced (default)


# ================================== Main Body ========================================
set.seed(random_seed)
# set a X11 window for data visualization when running from command line
#X11()

# ============= Data processing (read in features and labels etc.) =============
if (ml_model %in% c('lr','rf','et_R','et_sk','xgb','umap')) {
  
  # ------------------- In standard dataframe format --------------------
  # Read summary.txt which contains information about training and testing set
  # read_sum is defined in 'read_files.R'
  ls_set = read_sum(file.path(ads_dir,'summary.txt')) 
  # extract id
  vec_trainid = ls_set$train_set$id
  vec_testid = ls_set$test_set$id
  vec_allid = c(vec_trainid,vec_testid)
  
  # apply different features
  if (is.null(main_features)){
    stop('ERROR: Feature vector is empty. Select at least one type of features!!')
    
  } else {
    # initialize a df_joint
    df_joint = data.frame(id=vec_allid)
    df_histonly = NULL

    # read in pre-collected GCMC data
    gcmc_file = file.path(data_dir,gcmcname)
    df_gcmc = read.table(gcmc_file)

    # convert id in gcmc file to string format (7-18-2021)
    df_gcmc$id = as.character(df_gcmc$id)

    # joint data frame
    df_joint = df_joint %>%left_join(df_gcmc, by="id")
  }
  
  # always read in 2D histogram features
  # function defined in read_files.R
  ls_xy = read_hist_gcmc(hist_path = grid_dir,
                         tdhistname = tdhistname,
                         gcmc_file = gcmc_file,
                         idset = vec_allid,
                         nonzero_load = FALSE,
                         byrow = FALSE)
  
  # extract number of rows and columns of 2D histogram matrix for later conversion use
  hist_nrow = ls_xy$hist_nrow
  hist_ncol = ls_xy$hist_ncol
  
  # extract coordinates for heatmap plotting
  ener_coord = ls_xy$xcoord
  norm_coord = ls_xy$ycoord
  
  
  
  # use 2D histogram as features
  if ('2dhist' %in% main_features) {
    
    # since we already read in gcmc, we exclude loading here
    df_hist_no_loading = dplyr::select(ls_xy$df_xy,-c(loading))
    
    # dataframe with 2dhist and gcmc loading
    df_joint = df_joint %>% left_join(df_hist_no_loading, by = "id")
    
    # this dataframe is retained for plotting heatmap
    df_histonly = df_hist_no_loading %>% dplyr::select(.,-c(id))
    
    
  }
  
  
  # read in 1D persistent image features
  if ('persimg_1d' %in% main_features) {
    
    # function defined in read_files.R
    df_persimg1 = read_pers_image(file.path(ads_dir,'persimg_1D_all.csv'))
    
    # feature processing (e.g., remove zero variance column etc)
    # consistent for both training/testing set, function defined in 'data_process.R'
    #ls_x = remove_cols_x(df_persimg,meancut,stdcut)
    #df_persimg = ls_x$x
    
    # join to the main data frame
    df_joint = df_joint %>% left_join(df_persimg1, by = "id")
  }

  
  # read in 2D persistent image features
  if ('persimg_2d' %in% main_features) {
    
    # function defined in read_files.R
    df_persimg2 = read_pers_image(file.path(ads_dir,'persimg_2D_all.csv'))
    
    # feature processing (e.g., remove zero variance column etc)
    # consistent for both training/testing set, function defined in 'data_process.R'
    # ls_x = remove_cols_x(df_persimg,meancut,stdcut)
    # df_persimg = ls_x$x
    
    # join data frame
    df_joint = df_joint %>% left_join(df_persimg2, by = "id")
  }

  
  # read in basic structural features
  if ('textural' %in% main_features) {
    
    # check empty vector
    if (is.null(tex_features)) {
      stop('ERROR: empty tex_features vector for basic textural features!!')
    }
    
    # get textural properties
    # function defined in 'read_files.R'
    df_texprop = read_csv_texprop(os = os_current,
                                  extra_features = tex_features)
    
    # join to the main data frame
    df_joint = df_joint %>% left_join(df_texprop, by = "id")
  }


  # training data
  df_x_train_original = dplyr::filter(df_joint, df_joint$id %in% vec_trainid) %>% dplyr::select(.,-c(id,loading))
  vec_y_train = dplyr::filter(df_joint, df_joint$id %in% vec_trainid) %>% .$loading
  trainid = dplyr::filter(df_joint, df_joint$id %in% vec_trainid) %>% .$id
  
  # testing data
  df_x_test_original = dplyr::filter(df_joint, df_joint$id %in% vec_testid) %>% dplyr::select(.,-c(id,loading))
  vec_y_test = dplyr::filter(df_joint, df_joint$id %in% vec_testid) %>% .$loading
  testid = dplyr::filter(df_joint, df_joint$id %in% vec_testid) %>% .$id
  
  
  
  # feature scaling
  if (scaler == 'standardize') {
    
    # standardize with zero mean and unit variance, learn from training data only
    ls_std_train = standardize_x_fit(df_x_train_original)
    # standardized feature set for training data
    df_x_train = ls_std_train$std_x 
    
    # apply scalar to the testing data
    ls_std_test = standardize_x_transform(df_x_test_original,ls_std_train)
    df_x_test = ls_std_test$std_x
    
    
  } else if (scaler == 'normalize'){
    
    # normalize to [0,1]
    ls_norm_train = minmaxscale_x_fit(df_x_train_original)
    df_x_train  = ls_norm_train$norm_x
    # apply scalar to the testing data set
    ls_norm_test = minmaxscale_x_transform(df_x_test_original,ls_norm_train)
    df_x_test = ls_norm_test$norm_x
    
    
  } else if (scaler == 'none') {
    
    # retain original value
    df_x_train = df_x_train_original
    df_x_test  = df_x_test_original
    
  } else {
    stop("ERROR: invalid option for 'scaler' variable!!")
    
  }
  

  
  
  
} else if (ml_model %in% c('cnn','mlp','gs')) {
  
  # (4/22/2021) Currently only works for histogram and textural property features
  # ------------------- In array format ----------------------
  # Read summary.txt which contains information about training and testing set
  # read_sum is defined in 'read_files.R'
  ls_set = read_sum(file.path(ads_dir,'summary.txt')) 
  
  # extract id
  vec_trainid = ls_set$train_set$id
  vec_testid = ls_set$test_set$id
  vec_allid = c(vec_trainid,vec_testid)
  
  # prepare gcmc file directory
  gcmc_file = file.path(data_dir,gcmcname)
  
  # always first read 2D histogram data 
  # function defined in read_files.R
  ls_xy = read_hist_gcmc_mx(hist_path = grid_dir,
                            tdhistname = tdhistname,
                            gcmc_file = gcmc_file,
                            idset = vec_allid)
  # extract data
  ary_x_2dhist = ls_xy$x
  mx_y = ls_xy$y
  df_id = ls_xy$df_id
  hist_nrow = ls_xy$hist_nrow
  hist_ncol = ls_xy$hist_ncol
  # add a sequence id column 
  df_id = mutate(df_id,seq=c(1:nrow(df_id)))
  
  
  # feature processing
  if (ml_model %in% c('cnn')) {
    
    # pixel standardization
    # if (standardize) {
    #   ary_x = standardize_x_array4cnn(ary_x)
    # }
    
    # training data
    vec_seq_train = dplyr::filter(df_id, df_id$id %in% vec_trainid) %>% .$seq
    ary_x_train = ary_x[vec_seq_train,,]
    mx_y_train = matrix(mx_y[vec_seq_train,])
    trainid = dplyr::filter(df_id, df_id$id %in% vec_trainid) %>% .$id
    
    # testing data
    vec_seq_test = dplyr::filter(df_id, df_id$id %in% vec_testid) %>% .$seq
    ary_x_test = ary_x[vec_seq_test,,]
    mx_y_test = matrix(mx_y[vec_seq_test,])
    testid = dplyr::filter(df_id, df_id$id %in% vec_testid) %>% .$id
    
    # reshape the data set to suit Keras (image depth == 1 in our case)
    ary_x_train = tensorflow::array_reshape(ary_x_train, c(nrow(ary_x_train), hist_nrow, hist_ncol, 1))
    ary_x_test = tensorflow::array_reshape(ary_x_test, c(nrow(ary_x_test), hist_nrow, hist_ncol, 1))
    
    
  } else if (ml_model %in% c('mlp','gs')){
    
    # initialize feature array
    ary_x = array(numeric(),dim = c(nrow(ary_x_2dhist),0)) 
    
    
    # use 2D histogram as features
    if ('2dhist' %in% main_features) {
      
      # total number of features 
      n_dim = hist_nrow*hist_ncol
    
      # reshape the array data (flatten)
      ary_x = tensorflow::array_reshape(ary_x_2dhist, c(nrow(ary_x_2dhist),n_dim)) %>% cbind(ary_x)
      
    }
    

    # textural features
    if ( ('textural' %in% main_features) & (!is.null(tex_features)) ) {
      
      # read textural data
      # function defined in 'read_files.R'
      df_texprop_sel = read_csv_texprop(os_current,tex_features)
      
      # filter out useful textural properties 
      df_texprop_sel = df_id %>% left_join(df_texprop_sel, by="id")

      
      # extract names for extra features (including names for one-hot part)
      vec_colnames = colnames(df_texprop_sel)
      vec_featurenames = vec_colnames[!vec_colnames %in% c('id','seq')]
      
      # add extra features into the array
      for (icol in vec_featurenames) {
        ary_x = cbind(ary_x,df_texprop_sel[,icol])
      }
      
      
    } 
    
    # training data
    vec_seq_train = dplyr::filter(df_id, df_id$id %in% vec_trainid) %>% .$seq
    ary_x_train_original = ary_x[vec_seq_train,]
    mx_y_train = matrix(mx_y[vec_seq_train,])
    trainid = dplyr::filter(df_id, df_id$id %in% vec_trainid) %>% .$id
    
    
    # testing data
    vec_seq_test = dplyr::filter(df_id, df_id$id %in% vec_testid) %>% .$seq
    ary_x_test_original = ary_x[vec_seq_test,]
    mx_y_test = matrix(mx_y[vec_seq_test,])
    testid = dplyr::filter(df_id, df_id$id %in% vec_testid) %>% .$id
    
    

    # feature scaling
    if (scaler == 'standardize') {
      
      # standardize with zero mean and unit variance, learn from training data only
      ls_std_train = standardize_x_fit(data.frame(ary_x_train_original))
      # standardized feature set for training data
      ary_x_train = data.matrix(ls_std_train$std_x) 
      
      # apply scalar to the testing data
      ls_std_test = standardize_x_transform(data.frame(ary_x_test_original),ls_std_train)
      ary_x_test = data.matrix(ls_std_test$std_x)
      
      
    } else if (scaler == 'normalize'){
      
      # normalize to [0,1]
      ls_norm_train = minmaxscale_x_fit(data.frame(ary_x_train_original))
      ary_x_train  = data.matrix(ls_norm_train$norm_x)
      # apply scalar to the testing data set
      ls_norm_test = minmaxscale_x_transform(data.frame(ary_x_test_original),ls_norm_train)
      ary_x_test = data.matrix(ls_norm_test$norm_x)
      
      
    } else if (scaler == 'none') {
      
      # retain original value
      ary_x_train = ary_x_train_original
      ary_x_test  = ary_x_test_original
      
    } else {
      stop("ERROR: invalid option for 'scaler' variable!!")
      
    }
    
    

    
  } 
  
} 


# ============  Training and Testing of ML models  ===================

if (ml_model == 'lr') {

  #======================= Linear Regression ======================
  # initialize empty data frame for learning curve
  df_lcv = data.frame()
  
  # Learning curve 
  if (learning_curve) {
    
    # set up different number of training data
    n_traindata = seq(n_train_start,nrow(df_x_train),n_intv)

     # set up progress bar
    pb = txtProgressBar(min = 1, max = length(n_traindata), style = 3)
    pt = 0
    
    # loop 
    for (itrain in n_traindata) {
      
      # randomly sample
      rows = sample(nrow(df_x_train),itrain)
      
      # function defined in 'ml_functions.R' 
      ls_out = Linear_Regression(xtrain = df_x_train[rows,],
                                 ytrain = vec_y_train[rows],
                                 xtest = df_x_test,
                                 ytest = vec_y_test,
                                 ALPHA = DEFAULT_ALPHA)
      
      # extract rmse
      rmse_train = ls_out$train$eval_metric$rmse
      rmse_test  = ls_out$test$eval_metric$rmse
      
      # row bind to data frame
      df_tmp = data.frame(x=c(itrain),train=c(rmse_train),test=c(rmse_test))
      df_lcv = rbind(df_lcv,df_tmp)
      
      # progress bar
      pt = pt + 1 
      setTxtProgressBar(pb, pt)
    }
  } else {
    
    # standard LASSO
    ls_out = Linear_Regression(xtrain = df_x_train,
                               ytrain = vec_y_train,
                               xtest = df_x_test,
                               ytest = vec_y_test,
                               ALPHA = DEFAULT_ALPHA)
    
  }
  
  # collect all data to a list (for learning curve, only the last model will be saved here)
  ls_save = list(train = ls_out$train,
                 test  = ls_out$test,
                 lcv = df_lcv,
                 hist_nrow = hist_nrow,
                 hist_ncol = hist_ncol,
                 xcoord = ener_coord,
                 ycoord = norm_coord,
                 trainid = trainid,
                 testid = testid,
                 hist_only = df_histonly)
  
  # write results to RDS file
  write_rds(ls_save, paste0(output,'.rds'))
  
  
  
} else if (ml_model == 'rf'){
  
  # ==================== Random Forest =====================
  # initialize empty dataframe for learning curve
  df_lcv = data.frame()
  
  # Learning curve 
  if (learning_curve) {
    
    # set up different number of training data
    n_traindata = seq(n_train_start,nrow(df_x_train),n_intv)

    # set up progress bar
    pb = txtProgressBar(min = 1, max = length(n_traindata), style = 3)
    pt = 0
    
    # loop 
    for (itrain in n_traindata) {
      
      # randomly sample
      rows = sample(nrow(df_x_train),itrain)
      
      # function defined in 'ml_functions.R' 
      ls_out = Random_Forest(xtrain = df_x_train[rows,],
                             ytrain = vec_y_train[rows],
                             xtest = df_x_test,
                             ytest = vec_y_test,
                             ntree = n_tree,
                             nodesize = nsample_leafnode)
      
      # extract rmse
      rmse_train = ls_out$train$eval_metric$rmse
      rmse_test  = ls_out$test$eval_metric$rmse
      
      # row bind to data frame
      df_tmp = data.frame(x=c(itrain),train=c(rmse_train),test=c(rmse_test))
      df_lcv = rbind(df_lcv,df_tmp)
      
      # progress bar
      pt = pt + 1 
      setTxtProgressBar(pb, pt)
    }
  
  # No need for learning curve   
  } else {
    
    # standard Random Forest
    ls_out = Random_Forest(xtrain = df_x_train,
                           ytrain = vec_y_train,
                           xtest = df_x_test,
                           ytest = vec_y_test,
                           ntree = n_tree,
                           nodesize = nsample_leafnode)
    
  }
  
  # a list of data that is saved for later use
  ls_save = list(train = ls_out$train,
                 test  = ls_out$test,
                 lcv = df_lcv,
                 hist_nrow = hist_nrow,
                 hist_ncol = hist_ncol,
                 xcoord = ener_coord,
                 ycoord = norm_coord,
                 trainid = trainid,
                 testid = testid)
  
  # write results to RDS file
  write_rds(ls_save, paste0(output,'.rds'))
  
  
  
} else if (ml_model == 'cnn'){
  
  # ==================== Convolutional Neural Nets ========================
  # set random seed
  tensorflow::tf$random$set_seed(random_seed)
  
  input_shape = c(hist_nrow, hist_ncol, 1)
  
  # define model
  cnn_mod = keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                   padding = 'same', input_shape = input_shape) %>%
    #layer_max_pooling_2d(pool_size = c(2, 2),strides = 2) %>%
    #layer_conv_2d(filters = 32, kernel_size = c(5,5), activation = 'relu',
    #              padding = 'same') %>%
    #layer_dropout(rate = 0.25) %>%
    
    layer_conv_2d(filters = 64, kernel_size = c(5,5), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2),strides = 2) %>%
    #layer_conv_2d(filters = 32, kernel_size = c(5,5), activation = 'relu') %>%
    #layer_dropout(rate = 0.25) %>%
    
    layer_flatten() %>%
    layer_dense(units = 512, activation = 'relu') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 512, activation = 'relu') %>%
    layer_dense(units = 512, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'linear')
  
  # Compile model
  # optimizer option: adam, adagrad, rmsprop, sgd
  cnn_mod %>% compile(loss = "mse",
                      optimizer = optimizer_adam(lr = 0.0005))
  
  # Train model
  cnn_mod %>% fit(ary_x_train,
                  mx_y_train,
                  batch_size = 32,
                  epochs = 500,
                  validation_split = 0.2,
                  callbacks = list(callback_early_stopping(monitor = 'val_loss',
                                                           min_delta = 5, 
                                                           patience = 300,
                                                           restore_best_weights = TRUE),
                                   callback_reduce_lr_on_plateau(monitor = 'val_loss',
                                                                 patience = 50,
                                                                 factor = 0.5))
                                   
                  )
  
  # predict on training data
  vec_predy_train = cnn_mod %>% predict_on_batch(.,ary_x_train) %>% as.vector(.)
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric_train = eval_metric(as.vector(mx_y_train), vec_predy_train)
  
    # A list for the trained model
  ls_train = list(x = ary_x_train,
                  actu_y = as.vector(mx_y_train),
                  pred_y = vec_predy_train,
                  eval_metric = ls_metric_train)
  
  # evaluate on testing data
  vec_predy_test = cnn_mod %>% predict_on_batch(.,ary_x_test) %>% as.vector(.)
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric_test = eval_metric(as.vector(mx_y_test), vec_predy_test)
  
  # A list for the test
  ls_test = list(x = ary_x_test,
                 actu_y = as.vector(mx_y_test),
                 pred_y = vec_predy_test,
                 eval_metric = ls_metric_test)
  
  # a list of data that is saved for later use
  ls_save = list(train = ls_train,
                 test  = ls_test,
                 hist_nrow = hist_nrow,
                 hist_ncol = hist_ncol,
                 #xcoord = ener_coord,
                 #ycoord = norm_coord,
                 trainid = trainid,
                 testid = testid)
  
  # write results to RDS file
  write_rds(ls_save, paste0(output,'.rds'))
  
  # write TF model to file
  save_model_hdf5(cnn_mod,paste0(output,'.h5'))

  
} else if (ml_model == 'mlp') {
  
  # ==================== Neural Nets =========================
  # set random seed
  tensorflow::tf$random$set_seed(random_seed)
  
  input_dim = dim(ary_x_train)[2]

  # hyperparameters tunning
  if (tunehyperpara) {
    
    # define tunning flag
    FLAGS = flags(flag_integer('dense1',50),
                  flag_integer('dense2',50),
                  flag_integer('dense3',50),
                  flag_integer('dense4',50),
                  flag_numeric('dropout1',0.1),
                  flag_string('activ','relu'))
    
    
    # define model
    mlp_mod = keras_model_sequential() %>%
  
      layer_dense(units = FLAGS$dense1, activation = FLAGS$activ, input_shape = input_dim) %>%
      layer_dropout(rate = FLAGS$dropout1) %>%
      layer_dense(units = FLAGS$dense2, activation = FLAGS$activ) %>%
      # layer_dropout(rate = FLAGS$dropout2) %>%
      layer_dense(units = FLAGS$dense3, activation = FLAGS$activ) %>%
      layer_dense(units = FLAGS$dense4, activation = FLAGS$activ) %>%
      #layer_dropout(rate = 0.5) %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Compile model
    # optimizer option: adam, adagrad, rmsprop, sgd
    mlp_mod %>% compile(loss = "mse",
                        optimizer = optimizer_adam(lr = 0.0005)) 
                        #metrics = c(metric_mean_squared_error))
    
    # Train model
    history = mlp_mod %>% fit(ary_x_train,
                              mx_y_train,
                              batch_size = 32,
                              epochs = 2000,
                              validation_split = 0.2,
                              callbacks = list(callback_early_stopping(monitor = 'val_loss',
                                                                       min_delta = 5, 
                                                                       patience = 300,
                                                                       restore_best_weights = TRUE),
                                               callback_reduce_lr_on_plateau(monitor = 'val_loss',
                                                                             patience = 50,
                                                                             factor = 0.5)
                              ))
    
      
    
  } else {
    
    # Run model directly without tunning
    mlp_mod = keras_model_sequential() %>%
      
      layer_dense(units = 256, activation = 'relu', input_shape = input_dim) %>%
      layer_dropout(rate = 0.4) %>%
      layer_dense(units = 128, activation = 'relu') %>%
      layer_dense(units = 32, activation = 'relu') %>%
      # layer_dropout(rate = 0.1) %>%
      layer_dense(units = 32, activation = 'relu') %>%
      # layer_dense(units = 512, activation = 'relu') %>%
      #layer_dense(units = 64, activation = 'relu') %>%

      
      #layer_dropout(rate = 0.5) %>%
      layer_dense(units = 1, activation = 'linear')
    
    # Compile model
    # optimizer option: adam, adagrad, rmsprop, sgd
    mlp_mod %>% compile(loss = "mse",
                        optimizer = optimizer_adam(lr = 0.0005)) 
    #metrics = c(metric_mean_squared_error))
    
    # Train model
    mlp_mod %>% fit(ary_x_train,
                    mx_y_train,
                    batch_size = 32,
                    epochs = 2000,
                    validation_split = 0.2,
                    callbacks = list(callback_early_stopping(monitor = 'val_loss',
                                                             min_delta = 5,
                                                             patience = 300,
                                                             restore_best_weights = TRUE),
                                     callback_reduce_lr_on_plateau(monitor = 'val_loss',
                                                                   patience = 50,
                                                                   factor = 0.5)

                                     )
                    )
    
    # predict on training data
    vec_predy_train = mlp_mod %>% predict_on_batch(.,ary_x_train) %>% as.vector(.)
    
    # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
    ls_metric_train = eval_metric(as.vector(mx_y_train), vec_predy_train)
    
    # A list for the trained model
    ls_train = list(x = ary_x_train,
                    actu_y = as.vector(mx_y_train),
                    pred_y = vec_predy_train,
                    eval_metric = ls_metric_train)
    
    # evaluate on testing data
    vec_predy_test = mlp_mod %>% predict_on_batch(.,ary_x_test) %>% as.vector(.)
    
    # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
    ls_metric_test = eval_metric(as.vector(mx_y_test), vec_predy_test)
    
    # A list for the test
    ls_test = list(x = ary_x_test,
                   actu_y = as.vector(mx_y_test),
                   pred_y = vec_predy_test,
                   eval_metric = ls_metric_test)
    
    # a list of data that is saved for later use
    ls_save = list(train = ls_train,
                   test  = ls_test,
                   hist_nrow = hist_nrow,
                   hist_ncol = hist_ncol,
                   #xcoord = ener_coord,
                   #ycoord = norm_coord,
                   trainid = trainid,
                   testid = testid)
    
    # write results to RDS file
    write_rds(ls_save, paste0(output,'.rds'))
    
    # write TF models to files
    save_model_hdf5(mlp_mod,paste0(output,'.h5'))
    
    
  }

  
  
} else if (ml_model == 'et_R'){ 
  
  # ==================== Extra Trees (R version) =====================
  library("extraTrees")
  # initialize empty dataframe for learning curve
  df_lcv = data.frame()
  
  # Learning curve 
  if (learning_curve) {
    
    # set up different number of training data
    n_traindata = seq(n_train_start,nrow(df_x_train),n_intv)
    
    # set up progress bar
    pb = txtProgressBar(min = 1, max = length(n_traindata), style = 3)
    pt = 0
    
    # loop 
    for (itrain in n_traindata) {
      
      # randomly sample
      rows = sample(nrow(df_x_train),itrain)
      
      # function defined in 'ml_functions.R' 
      ls_out = Extra_Trees(xtrain = df_x_train[rows,],
                           ytrain = vec_y_train[rows],
                           xtest = df_x_test,
                           ytest = vec_y_test,
                           ntree = n_tree,
                           nodesize = nsample_leafnode,
                           nfeature = n_feature,
                           tunerandomcut = iftunerandomcut)
      
      # extract rmse
      rmse_train = ls_out$train$eval_metric$rmse
      rmse_test  = ls_out$test$eval_metric$rmse
      
      # row bind to data frame
      df_tmp = data.frame(x=c(itrain),train=c(rmse_train),test=c(rmse_test))
      df_lcv = rbind(df_lcv,df_tmp)
      
      # progress bar
      pt = pt + 1 
      setTxtProgressBar(pb, pt)
    }
  } else {
    
    # standard Extra Tree
    ls_out = Extra_Trees(xtrain = df_x_train,
                         ytrain = vec_y_train,
                         xtest = df_x_test,
                         ytest = vec_y_test,
                         ntree = n_tree,
                         nodesize = nsample_leafnode,
                         nfeature = n_feature,
                         tunerandomcut = iftunerandomcut)
    
    
  }
    
  # a list of data that is saved for later use
  ls_save = list(train = ls_out$train,
                 test  = ls_out$test,
                 lcv = df_lcv,
                 hist_nrow = hist_nrow,
                 hist_ncol = hist_ncol,
                 xcoord = ener_coord,
                 ycoord = norm_coord,
                 trainid = trainid,
                 testid = testid)
  
  # write results to RDS file
  write_rds(ls_save, paste0(output,'.rds'))

} else if (ml_model == 'et_sk') {
  
  # ==================== Extra Trees (Python version) =================
  # Using the sklearn package in Python
  library("reticulate")
  # load the native python version
  reticulate::use_python("C:/Users/khshi/AppData/Local/Programs/Python/Python37",required = TRUE)
  
  # import sklearn package
  sklearn = reticulate::import('sklearn.ensemble')
  
  # Initialize Extra Tree Regressor
  etr = sklearn$ExtraTreesRegressor(n_estimators = as.integer(n_tree),
                                    criterion = "mse", # consistent with R version
                                    min_samples_leaf = as.integer(nsample_leafnode),
                                    max_features = "auto",
                                    bootstrap = FALSE, # the whole dataset is used to build each tree
                                    oob_score = FALSE, # flag to use OOB sample to estiamte R2 for unseen data
                                    n_jobs = as.integer(6), # use six cores for parallel running
                                    random_state = as.integer(random_seed))
  
  # initialize empty dataframe for learning curve
  df_lcv = data.frame()
  ls_train = list()
  ls_test = list()
  
  # Learning curve 
  if (learning_curve) {
    
    # set up different number of training data
    n_traindata = seq(n_train_start,nrow(df_x_train),n_intv)
    
    # set up progress bar
    pb = txtProgressBar(min = 1, max = length(n_traindata), style = 3)
    pt = 0
    
    # loop 
    for (itrain in n_traindata) {
      
      # randomly sample
      rows = sample(nrow(df_x_train),itrain)
      
      # regression
      et_mod = etr$fit(df_x_train[rows,], vec_y_train[rows])
      
      # predict
      vec_y_pred = et_mod$predict(df_x_test)
      ls_metric = eval_metric(vec_y_test, vec_y_pred)
      rmse_test  = ls_metric$rmse
      
      # row bind to data frame
      df_tmp = data.frame(x=c(itrain),train=NA,test=c(rmse_test))
      df_lcv = rbind(df_lcv,df_tmp)
      
      # progress bar
      pt = pt + 1 
      setTxtProgressBar(pb, pt)
    }
    
  } else {
    
    # regress
    et_mod = etr$fit(df_x_train, vec_y_train)
    
    # return a list for the trained model
    ls_train = list(mod = NULL,
                    x = df_x_train,
                    actu_y = vec_y_train,
                    pred_y = NULL,
                    eval_metric = NULL)
    
    # prediction for testing data
    vec_y_pred = et_mod$predict(df_x_test)
    
    # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
    ls_metric = eval_metric(vec_y_test, vec_y_pred)
    
    # return a list for the trained model
    ls_test = list(x = df_x_test,
                   actu_y = vec_y_test,
                   pred_y = vec_y_pred,
                   eval_metric = ls_metric)
    

  }

  
  # a list of data that is saved for later use
  ls_save = list(train = ls_train,
                 test  = ls_test,
                 lcv = df_lcv,
                 hist_nrow = hist_nrow,
                 hist_ncol = hist_ncol,
                 xcoord = ener_coord,
                 ycoord = norm_coord,
                 trainid = trainid,
                 testid = testid)
  
  # write results to RDS file
  write_rds(ls_save, paste0(output,'.rds'))
  
  # write python models to pickle files
  reticulate::py_save_object(et_mod,paste0(output,'.pk'), pickle = "pickle")
  
  
} else if (ml_model == 'xgb') {
  
  # ================== Gradient Boosting Tree Method =================
  library("xgboost")
  
  # initialize empty dataframe for learning curve
  df_lcv = data.frame()
  
  # Learning curve 
  if (learning_curve) {
    
    # set up different number of training data
    n_traindata = seq(n_train_start,nrow(df_x_train),n_intv)
    
    # set up progress bar
    pb = txtProgressBar(min = 1, max = length(n_traindata), style = 3)
    pt = 0
    
    # loop 
    for (itrain in n_traindata) {
      
      # randomly sample
      rows = sample(nrow(df_x_train),itrain)
      
      # function defined in 'ml_functions.R' 
      ls_out = XGradient_Boosting(xtrain = df_x_train[rows,],
                                  ytrain = vec_y_train[rows],
                                  xtest = df_x_test,
                                  ytest = vec_y_test,
                                  xgbGrid = xgbGrid)
      
      # extract rmse
      rmse_train = ls_out$train$eval_metric$rmse
      rmse_test  = ls_out$test$eval_metric$rmse
      
      # row bind to data frame
      df_tmp = data.frame(x=c(itrain),train=c(rmse_train),test=c(rmse_test))
      df_lcv = rbind(df_lcv,df_tmp)
      
      # progress bar
      pt = pt + 1 
      setTxtProgressBar(pb, pt)
    }
  } else {
    
    # standard gradient boosting tree, defined in 'ml_functions.R'
    ls_out = XGradient_Boosting(xtrain = df_x_train,
                                ytrain = vec_y_train,
                                xtest = df_x_test,
                                ytest = vec_y_test,
                                xgbGrid = xgbGrid)
    
  }
  
  # a list of data that is saved for later use
  ls_save = list(train = ls_out$train,
                 test  = ls_out$test,
                 lcv = df_lcv,
                 hist_nrow = hist_nrow,
                 hist_ncol = hist_ncol,
                 xcoord = ener_coord,
                 ycoord = norm_coord,
                 trainid = trainid,
                 testid = testid)
  
  # write results to RDS file
  write_rds(ls_save, paste0(output,'.rds'))
  
  
  
  
} else if (ml_model == 'gs') {
  
  # ================== Stacking method (needs to be updated, double-check following code, 3/5/2021) ======================
  # set random seed
  tensorflow::tf$random$set_seed(random_seed)
  
  # read in trained ML model for ethane at different pressure
  mod1 = load_model_hdf5('DNN_eth4_14x5_optimized.h5')
  mod2 = load_model_hdf5('DNN_eth20_14x5.h5')
  mod3 = load_model_hdf5('DNN_eth40_14x5.h5')
  
  # dimension of features
  tddim = hist_nrow*hist_ncol
  
  # reshape the training and testing set
  ary_x_train = array_reshape(ary_x_train, c(nrow(ary_x_train), tddim))
  ary_x_test  = array_reshape(ary_x_test, c(nrow(ary_x_test), tddim))
  
  # extract pre-trained ML models
  vec_predy1 = mod1 %>% predict_on_batch(.,ary_x_train) %>% as.vector(.)
  vec_predy2 = mod2 %>% predict_on_batch(.,ary_x_train) %>% as.vector(.)
  vec_predy3 = mod3 %>% predict_on_batch(.,ary_x_train) %>% as.vector(.)
  
  # construct new features for DNN
  ary_x_train_meta = array(0, dim=c(nrow(ary_x_train),3))
  ary_x_train_meta[] = c(vec_predy1,vec_predy2,vec_predy3)
  ary_x_train_meta = abind::abind(ary_x_train_meta,ary_x_train)
  # prepare test data
  vec_predy1 = mod1 %>% predict_on_batch(.,ary_x_test) %>% as.vector(.)
  vec_predy2 = mod2 %>% predict_on_batch(.,ary_x_test) %>% as.vector(.)
  vec_predy3 = mod3 %>% predict_on_batch(.,ary_x_test) %>% as.vector(.)
  
  # construct new features for DNN
  ary_x_test_meta = array(0, dim=c(nrow(ary_x_test),3))
  ary_x_test_meta[] = c(vec_predy1,vec_predy2,vec_predy3)
  ary_x_test_meta = abind::abind(ary_x_test_meta,ary_x_test)
  
  # Run model directly without tunning
  dnn_mod = keras_model_sequential() %>%
    
    layer_dense(units = 1024, activation = 'relu', input_shape = c(tddim+3)) %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 1024, activation = 'relu') %>%
    layer_dense(units = 512, activation = 'relu') %>%

    
    #layer_dropout(rate = 0.5) %>%
    layer_dense(units = 1, activation = 'linear')
  
  # Compile model
  # optimizer option: adam, adagrad, rmsprop, sgd
  dnn_mod %>% compile(loss = "mse",
                      optimizer = optimizer_adam(lr = 0.0005)) 
  #metrics = c(metric_mean_squared_error))
  
  # Train model
  dnn_mod %>% fit(ary_x_train_meta,
                  mx_y_train,
                  batch_size = 32,
                  epochs = 1000,
                  validation_split = 0.2,
                  callbacks = list(callback_early_stopping(monitor = 'val_loss',
                                                           min_delta = 5, 
                                                           patience = 300,
                                                           restore_best_weights = TRUE),
                                   callback_reduce_lr_on_plateau(monitor = 'val_loss',
                                                                 patience = 50,
                                                                 factor = 0.5)
                                   
                  ))
  
  # predict on training data
  vec_predy_train = dnn_mod %>% predict_on_batch(.,ary_x_train_meta) %>% as.vector(.)
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric_train = eval_metric(as.vector(mx_y_train), vec_predy_train)
  
  # A list for the trained model
  ls_train = list(x = ary_x_train_meta,
                  actu_y = as.vector(mx_y_train),
                  pred_y = vec_predy_train,
                  eval_metric = ls_metric_train)
  
  # evaluate on testing data
  vec_predy_test = dnn_mod %>% predict_on_batch(.,ary_x_test_meta) %>% as.vector(.)
  
  # calculate mean absolute error (MAE), root mean squared error (RMSE), R-squared
  ls_metric_test = eval_metric(as.vector(mx_y_test), vec_predy_test)
  
  # A list for the test
  ls_test = list(x = ary_x_test_meta,
                 actu_y = as.vector(mx_y_test),
                 pred_y = vec_predy_test,
                 eval_metric = ls_metric_test)
  
  # a list of data that is saved for later use
  ls_save = list(train = ls_train,
                 test  = ls_test,
                 hist_nrow = hist_nrow,
                 hist_ncol = hist_ncol,
                 #xcoord = ener_coord,
                 #ycoord = norm_coord,
                 trainid = trainid,
                 testid = testid)
  
  # write results to RDS file
  write_rds(ls_save, paste0(output,'.rds'))
  
  # write TF models to files
  save_model_hdf5(dnn_mod,paste0(output,'.h5'))
  
  
} else if (ml_model == 'umap') {
  
  # ====== Uniform Manifold Approximation and Projection (UMAP) ======== 
  # Two versions of implementations are available
  
  if (umap_version == 'R') {
    
    # https://cran.r-project.org/web/packages/umap/vignettes/umap.html
    # custom configuration 
    umap_config = umap.defaults
    umap_config$n_neighbors = umap_neighbors
    umap_config$n_epochs = umap_epochs
    umap_config$min_dist = umap_min_dist
    umap_config$random_state = random_seed
    
    # perform UMAP dimensional reduction
    umapout = umap(df_joint %>% dplyr::select(.,-c(id,loading)),
                   config = umap_config)
  
    ls_save = list(obj = umapout,
                   df_trans = data.frame(umapout$layout),
                   df_xy = df_joint)
    
  } else if (umap_version == 'Py'){
    # Using the original python UMAP code
    library("reticulate")
    # load the native python version
    reticulate::use_python("C:/Users/khshi/AppData/Local/Programs/Python/Python37",required = TRUE)
    
    # main body
    umap_py = reticulate::import('umap')
    # configure UMAP
    reducer = umap_py$UMAP(random_state = as.integer(random_seed),
                           min_dist = umap_min_dist,
                           n_epochs = as.integer(umap_epochs),
                           n_neighbors = as.integer(umap_neighbors),
                           # target metric for regression; for classification use default
                           target_metric = 'euclidean',
                           target_weight= umap_target_weight)
    
    # fit data
    umapout = reducer$fit_transform(df_joint %>% dplyr::select(.,-c(id,loading)),
                                    y = df_joint %>% dplyr::select(.,c(loading)))
    
    ls_save = list(obj = NA,
                   df_trans = data.frame(umapout),
                   df_xy = df_joint %>% dplyr::select(.,c(id,loading)))
    
  }
  
  
  # write to file
  write_rds(ls_save, paste0(output, '.rds'))
  
} else {
  
  stop('ERROR: NO ML MODEL IS SPECIFIED!')
  
  
}



# Sleep until the X11 windows are closed 
#while(names(dev.cur()) !='null device') Sys.sleep(1)















