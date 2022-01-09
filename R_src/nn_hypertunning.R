source('package_verification.R')
runs = tuning_run("main_ml_2dhist.R",
                  #sample = 0.5,  # only try half of the combinations of trial hyperparameters
                  flags = list(dense1 = c(1024), 
                               dense2 = c(1024),
                               dense3 = c(1024),
                               dropout1 = c(0.1,0.3,0.5),
                               dropout2 = c(0.1,0.3,0.5)
                                                   ))