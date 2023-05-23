# ===========================================
# Functions for manipulating/processing data
# ===========================================

source("package_verification.R")



# ==================== Simplify features by removing zero mean/var columns =========================
remove_cols_x = function(df_x, meancut, stdcut) {
  
  # ======================================================================
  # 3/2/2021
  # Output: Return a list containing a data frame of simplified features with reduced dimensionality
  # Input:
  # df_x     - a full data frame containing both features and labels
  # meancut  - when colmean <= meancut, this column is removed
  # varcut   - when colstd  <= stdcut, this column is removed
  # ======================================================================
  
  # pick out columns with small std dev
  vec_cols_remstd = df_x %>%
    summarize_all(sd) %>% 
    gather("bin", "sd") %>% 
    dplyr::filter(sd <= stdcut) %>% 
    .$bin
  
  # pick out columns with small means
  vec_cols_remmean = df_x %>%
    summarize_all(mean) %>% 
    gather("bin", "mean") %>% 
    dplyr::filter(mean <= meancut) %>% 
    .$bin
  
  # removed columns
  vec_removed_cols = dplyr::union(vec_cols_remstd,vec_cols_remmean)
  
  # remove columns
  df_x_new =  df_x[,!(names(df_x) %in% vec_removed_cols)]
  
  
  list(x = df_x_new,
       x_orig = df_x,
       removed_cols = vec_removed_cols)
}



# ==================== One-hot encoding of topology =========================
one_hot_categorical = function(vec_input) {
  
  # ======================================================================
  # Added on 3/8/2021
  # Output: Return a matrix containing the one-hot encoding of original 
  #         categorical data
  # Input:
  # vec_input     - a vector of input data
  # ======================================================================
  
  # extract unique categories from input
  unique_cat = unique(vec_input)
  
  # prepare a one-hot vector matrix
  mx_one_hot = matrix(0,nrow = length(vec_input),ncol = length(unique_cat))
  
  pt = 1
  # assign categorical data to matrix 
  for (idata in vec_input) {
    
    icol = which(unique_cat == idata)
    mx_one_hot[pt,icol] = 1
    pt = pt + 1
    
  }
  
  mx_one_hot
}





# ==================== Standardize the data set =========================
standardize_x_fit = function(df_x) {
  
  # ======================================================================
  # NOTE: this function is kind of redundant for glmnet because it will standardize the x by default
  #       The only minor difference might be sd() function used here is divided by (n-1), but standardization
  #       in glmnet is divided by n.
  #       Base function 'scale' did the same thing as this function
  #
  # Output: Return a list of standardized x data (data frame), 
  #         meanz (data frame), varz (data frame)
  # Input:
  # df_x     - data frame of flatted 2D histogram (training data only)
  # ======================================================================
  
  df_x_orig = df_x
  
  if (('topo' %in% names(df_x_orig)) & (length(names(df_x_orig)) ==1)) {
    # only topological code
    # function 'one_hot_categorical' is defined in 'data_process.R'
    mx_top = df_x$topo %>% one_hot_categorical()
    # to avoid default column names starting with 'X'
    colnames(mx_top) = colnames(mx_top,do.NULL = F,prefix = 't')
    df_x_std = data.frame(mx_top)
    df_meanz = NULL
    df_varz = NULL
    
  } else if (('topo' %in% names(df_x_orig)) & (length(names(df_x_orig)) >1)) {
    # topological code + other numerical features
    # function 'one_hot_categorical' is defined in 'data_process.R'
    mx_top = df_x$topo %>% one_hot_categorical()
    colnames(mx_top) = colnames(mx_top,do.NULL = F,prefix = 't')
    df_x = df_x[,which( !(names(df_x) %in% c('topo')))]
    df_meanz = df_x %>% summarize_all(mean)
    df_varz = df_x %>% summarize_all(sd)
    # Apply standardization procedures
    df_x_std = (df_x - df_meanz[rep(1,times=nrow(df_x)),]) / (df_varz[rep(1,times=nrow(df_x)),])
    df_x_std = cbind(df_x_std,mx_top)
    
  } else {
    # numerical feature only
    df_meanz = df_x %>% summarize_all(mean)
    df_varz = df_x %>% summarize_all(sd)
    # Apply standardization procedures
    df_x_std = (df_x - df_meanz[rep(1,times=nrow(df_x)),]) / (df_varz[rep(1,times=nrow(df_x)),])
    
  }
  
  list(
    std_x = df_x_std,
    df_x = df_x_orig,
    meanz = df_meanz,
    stdz = df_varz
  )
}


# ============== Standardize the data set by applying known scalar ==================
standardize_x_transform = function(df_x,ls_std) {
  
  # ======================================================================
  # Input:  
  # df_x     - testing data (data frame)
  # ls_std   - list output from standardize_x_fit function
  #
  # Output: Return a list of standardized x data (data frame)
  # ======================================================================
  
  df_x_orig = df_x
  
  df_meanz = ls_std$meanz
  df_varz = ls_std$stdz
  
  if (('topo' %in% names(df_x_orig)) & (length(names(df_x_orig)) ==1)) {
    # only topological code
    # function 'one_hot_categorical' is defined in 'data_process.R'
    mx_top = df_x$topo %>% one_hot_categorical()
    # to avoid default column names starting with 'X'
    colnames(mx_top) = colnames(mx_top,do.NULL = F,prefix = 't')
    df_x_std = data.frame(mx_top)

    
  } else if (('topo' %in% names(df_x_orig)) & (length(names(df_x_orig)) >1)) {
    # topological code + other numerical features
    # function 'one_hot_categorical' is defined in 'data_process.R'
    mx_top = df_x$topo %>% one_hot_categorical()
    colnames(mx_top) = colnames(mx_top,do.NULL = F,prefix = 't')
    df_x = df_x[,which( !(names(df_x) %in% c('topo')))]

    # Apply standardization procedures
    df_x_std = (df_x - df_meanz[rep(1,times=nrow(df_x)),]) / (df_varz[rep(1,times=nrow(df_x)),])
    df_x_std = cbind(df_x_std,mx_top)
    
  } else {

    # Apply standardization procedures
    df_x_std = (df_x - df_meanz[rep(1,times=nrow(df_x)),]) / (df_varz[rep(1,times=nrow(df_x)),])
    
  }
  
  list(
    std_x = df_x_std,
    df_x = df_x_orig,
    meanz = df_meanz,
    stdz = df_varz
  )
}



# =========== Normalize the data frame ==============
minmaxscale_x_fit = function(df_x) {
  
  # ======================================================================
  # Output: Return a normalized data frame
  # ======================================================================
  
  df_x_orig = df_x
  
  if (('topo' %in% names(df_x_orig)) & (length(names(df_x_orig)) ==1)) {
    # only topological code
    # function 'one_hot_categorical' is defined in 'data_process.R'
    mx_top = df_x$topo %>% one_hot_categorical()
    colnames(mx_top) = colnames(mx_top,do.NULL = F,prefix = 't')
    df_x_scaled = data.frame(mx_top)
    df_min = NULL
    df_max = NULL
    
  } else if (('topo' %in% names(df_x_orig)) & (length(names(df_x_orig)) >1)) {
    # topological code + other numerical features
    # function 'one_hot_categorical' is defined in 'data_process.R'
    mx_top = df_x$topo %>% one_hot_categorical()
    colnames(mx_top) = colnames(mx_top,do.NULL = F,prefix = 't')
    df_x = df_x[,which( !(names(df_x) %in% c('topo')))]
    # extract
    df_min = df_x %>% summarize_all(min)
    df_max = df_x %>% summarize_all(max)
    df_dif = df_max - df_min
    # normalized matrix
    df_x_scaled = (df_x - df_min[rep(1,times=nrow(df_x)),]) / (df_dif[rep(1,times=nrow(df_x)),])
    df_x_scaled = cbind(df_x_scaled,mx_top)
    
  } else {
    # numerical feature only
    df_min = df_x %>% summarize_all(min)
    df_max = df_x %>% summarize_all(max)
    df_dif = df_max - df_min
    # normalized matrix
    df_x_scaled = (df_x - df_min[rep(1,times=nrow(df_x)),]) / (df_dif[rep(1,times=nrow(df_x)),])
    
  }
  
  list(
    norm_x = df_x_scaled,
    df_x = df_x_orig,
    min = df_min,
    max = df_max
  )
  
}


# =========== Normalize the data frame ==============
minmaxscale_x_transform = function(df_x,ls_norm) {
  
  # ======================================================================
  # Output: Return a normalized data frame
  # ======================================================================
  
  df_x_orig = df_x
  
  df_min = ls_norm$min
  df_max = ls_norm$max
  df_dif = df_max - df_min
  
  if (('topo' %in% names(df_x_orig)) & (length(names(df_x_orig)) ==1)) {
    # only topological code
    # function 'one_hot_categorical' is defined in 'data_process.R'
    mx_top = df_x$topo %>% one_hot_categorical()
    colnames(mx_top) = colnames(mx_top,do.NULL = F,prefix = 't')
    df_x_scaled = data.frame(mx_top)

    
  } else if (('topo' %in% names(df_x_orig)) & (length(names(df_x_orig)) >1)) {
    # topological code + other numerical features
    # function 'one_hot_categorical' is defined in 'data_process.R'
    mx_top = df_x$topo %>% one_hot_categorical()
    colnames(mx_top) = colnames(mx_top,do.NULL = F,prefix = 't')
    df_x = df_x[,which( !(names(df_x) %in% c('topo')))]
    
    # normalized matrix
    df_x_scaled = (df_x - df_min[rep(1,times=nrow(df_x)),]) / (df_dif[rep(1,times=nrow(df_x)),])
    df_x_scaled = cbind(df_x_scaled,mx_top)
    
  } else {
    # numerical feature only
    # normalized matrix
    df_x_scaled = (df_x - df_min[rep(1,times=nrow(df_x)),]) / (df_dif[rep(1,times=nrow(df_x)),])
    
  }
  
  list(
    norm_x = df_x_scaled,
    df_x = df_x_orig,
    min = df_min,
    max = df_max
  )
  
}





# =========== Standardize the array for convoluted neural nets with zero mean and unit variance ==============
# standardize_x_array4cnn = function(ary_x) {
# 
#   # ======================================================================
#   # Output: Return an array with data standardized
#   # Input:
#   # ary_x     - array of x for CNN to be standardized
#   # ======================================================================
#   
#   # mean for all images (to be consistent throughout the entire data set)
#   meanx = mean(ary_x[,,,1])
#   # standard deviation (divided by n-1)
#   sdx = sd(ary_x[,,,1])
#   
#   # standardize matrix
#   ary_x[,,,1] = (ary_x[,,,1]-meanx)/sdx
# 
#   ary_x
# 
# }



# =========== Standardize the array for deep neural nets with z-score ==============
standardize_x_array4dnn = function(ary_x,mean = NULL,sd=NULL) {
  
  # ======================================================================
  # Output: Return an array with data standardized with zero mean and unit variance
  # Input:
  # ary_x     - array of x for DNN to be standardized 
  # mean      - mean of each columns. If NULL, it will be calculated from the current data set
  # sd        - standard deviation of columns. If NULL, it will be calculated from the current data
  # ======================================================================
  
  if (is.null(mean) & is.null(sd)) {
    
    vec_meanx = apply(ary_x, c(2),mean)    
    vec_sdx = apply(ary_x,c(2),sd)
  
  } else {
    
    vec_meanx = mean
    vec_sdx = sd
    
  }
  
  # keep columns with zero variance (this will not affect final results, neither the interpretation of model)
  #zero_var = which(vec_sdx==0)
  
  #vec_meanx = vec_meanx[-zero_var]
  #vec_sdx = vec_sdx[-zero_var]
  #ary_x = ary_x[,-zero_var]
  # standardize matrix
  ary_x = sweep(ary_x,2,vec_meanx) %>% sweep(.,2,vec_sdx,'/')

  # set NA to zero
  ary_x[is.na(ary_x)] = 0 
  
  list(x = ary_x,
       mean = vec_meanx,
       sd = vec_sdx)
  
}



# # ==================== Take log of the array  =========================
# log_x_array4cnn = function(ary_x,depth = 1) {
#   
#   # ======================================================================
#   # Output: Return an array with data
#   # Input:
#   # ary_x     - array of x for CNN
#   # depth     - depth of the image
#   # ======================================================================
#   
#   # loop over each image
#   for (irow in c(1:nrow(ary_x))) {
#     # loop over image depth
#     for (idim in c(1:depth)) {
#       
#       ary_x[irow,,,idim] = - log(ary_x[irow,,,idim])
#       ary_x[is.infinite(ary_x)] = 0  
#       
#     }
#   }
#   
#   # loop over each image
#   for (irow in c(1:nrow(ary_x))) {
#     # loop over image depth
#     for (idim in c(1:depth)) {
#       # mean
#       meanx = mean(ary_x[irow,,,idim])
#       # standard deviation (divided by n-1)
#       sdx = sd(ary_x[irow,,,idim])
#       # normalized matrix
#       ary_x[irow,,,idim] = (ary_x[irow,,,idim]-meanx)/sdx
#       
#     }
#   }
#   
#   ary_x
#   
# }



# =========== Normalize the array for deep neural nets ==============
minmaxscale_x_array4dnn = function(ary_x) {

  # ======================================================================
  # Output: Return a normalized array
  # Input:
  # ary_x     - array of x for DNN to be normalized
  # ======================================================================

  # extract
  vec_min = apply(ary_x,c(2),min)
  vec_max = apply(ary_x,c(2),max)
  vec_dif = vec_max - vec_min
  
  #zero_col = which(vec_dif==0)
  
  #vec_dif = vec_dif[-zero_col]
  #ary_x = ary_x[,-zero_col]
  
  # normalized matrix
  ary_x_scaled = sweep(ary_x,2,vec_min) %>% sweep(.,2,vec_dif,'/')

  ary_x_scaled

}



# ==================== Normalize y value  =========================
minmaxscale_y = function(mx_y,vec_minmax=NULL,inverse=FALSE) {
  
  # ======================================================================
  # Output: Return a list 
  # Input:
  # mx_y  -  matrix of y (targeted value)
  # vec_minmax - vector of scaling parameters: minimum and maximum
  # inverse  -  logic value determining normalize or denormalize
  # ======================================================================
  
  vec_y = as.vector(mx_y)

  
  if (inverse) {
    
    miny = vec_minmax[1]
    maxy = vec_minmax[2]
    
    denorm_y = vec_y * (maxy - miny) + miny
    
    ls_scale = list(y = denorm_y,
                    min = miny,
                    max = maxy)
    
  } else {
    
    # normalize y using min_max_scale
    miny = min(vec_y)
    maxy = max(vec_y)
    
    norm_y = (vec_y - miny)/(maxy - miny)
    
    ls_scale = list(y = norm_y,
                    min = miny,
                    max = maxy)
  }
  
  
  ls_scale
 
}



# =========== Normalize the array for deep neural nets ==============
polyfeatures_array4dnn = function(ary_x,degree=2) {
  
  # ======================================================================
  # Output: Return a new array of features containing higher order features and also
  #           preserves the original feature 
  # Input:
  # ary_x     - array of x for DNN 
  # degree    - order of features
  # ======================================================================
  
  ary_x_poly = cbind(ary_x,ary_x^degree)
  
  ary_x_poly
  
  
}




# ============== Calculate evaluation metrics for a trained model ==================
eval_metric = function(vec_orig, vec_pred){
  
  # ======================================================================
  # Output: Return a list of evaluation metrics: mean squre error (mse), root mean 
  #         square error (rmse), mean absolute error (mae), mean absolute percentage error (mape),
  #         and R-squared (R2).
  # Input:
  # vec_orig    - actual value of y
  # vec_pred    - predicted value of y
  # ======================================================================
  
  d = vec_orig - vec_pred
  mse = mean((d)^2)
  mae = mean(abs(d))
  rmse = sqrt(mse)
  # coefficient of determination
  R2 = 1-(sum((d)^2)/sum((vec_orig-mean(vec_orig))^2))
  # calculate mean absolute percentage error (setting some cutoff)
  # indx_valid = which(vec_orig > 1)
  # The MAPE here is slightly different, we use predicted value as denominator and it has two advantages:
  #  1). avoid zero denominator (given that the probability of getting zero prediction is very rare)
  #  2). it tells use the expected error for a given prediction
  mape = mean(abs(d/vec_pred)) * 100 
  #mape = mean(abs(d[indx_valid]/vec_orig[indx_valid])) * 100 
  
  list(diff = d,
       mse = mse,
       mae = mae,
       rmse = rmse,
       mape = mape,
       R2 = R2)
  
}



















