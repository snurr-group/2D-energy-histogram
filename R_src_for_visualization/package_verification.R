# Install missing packages that are required
# To install packages on Quest, you need to install them by running the 
#   R command line console (see https://kb.northwestern.edu/page.php?id=76569)
#options(java.parameters = "-Xmx4g")


packages = c(# PLOTTING PACKAGES
             "plotly","cowplot", "ggplot2","ggrepel",
             "viridis","wesanderson","RColorBrewer","gplots", 
             "grid", "hexbin", "ggExtra", "MASS", 
             
             # DATA ANALYSIS PACKAGES
             "dplyr", "stringr", "readr","tidyr",
             "tidyverse", "R.utils", 
             "manipulate", "slam", "purrr", "pracma",
             #"magrittr", "testthat",  
             
             # FILE RELATED PACKAGES
             "openxlsx", "readxl", "Rpdb","jsonlite",
             
             # MACHINE LEARNING PACKAGES
             "glmnet", "caret", "keras","tfruns","tfestimators",
             "randomForest","umap")


package.check <- lapply(packages, FUN = function(x) {
  
    if (!require(x, character.only = TRUE)) {
      
        install.packages(x, dependencies = TRUE)
        library(x, character.only = TRUE)
      
    }
})



