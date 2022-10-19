source('package_verification.R')
runs = tuning_run("main_ml_2dhist.R", flags = list(dense1 = c(128,256),
                                                   dense2 = c(32,64,128),
                                                   dense3 = c(32,64,128),
                                                   dense4 = c(32,64,128),
                                                   dropout1 = c(0.2, 0.3, 0.4),
                                                   activ = c('relu','sigmoid'))  )