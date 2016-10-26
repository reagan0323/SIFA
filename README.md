# SIFA
Supervised Integrated Factor Analysis
Matlab Codes for "Incorporating Covariates into Integrated Factor Analysis for Multi-View Data" by Gen Li and Sungkyu Jung (2016)

The folder contains all necessary functions for fitting SIFA models under different conditions. In particular, SIFA_A.m can be used to fit SIFA model under the general conditions with linear covariate functions. SIFA_A_np.m addresses nonparametric estimation of univariate covariate functions. SIFA_B.m can be used to fit SIFA model under the orthogonal conditions with linear covariate functions. SIFA_B_np.m addresses nonparametric estimation of univariate covariate functions. 

The two m-files contain simulation examples for model fitting and LCV rank estimation. All documents are well commented and ready to use.

The preprocessed GTEx gene expression data and covariates are contained in GTEx_data.mat. 
The preprocessed Berkeley growth velocity data are contained in Growth_data.mat.

Contact: Gen Li, gl2521@cumc.columbia.edu
