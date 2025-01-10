This contains the code to run the debias and the simulation described in the paper at https://arxiv.org/abs/2307.04527.

You must install, or have installed CVX and then include it's path in this.
%CVX:  https://github.com/cvxr/CVX/releases
%https://cvxr.com/cvx/download/

To run this for quick epochs, run:
debiassim_bigloop_split_v2025a2.m.  

Default settings runs a quick test with only 2 epochs of training.  The default simulation and neural network are a bit different than what is in the paper.

To do a long run, set quick_test = 0 inside the code;
To redo the paper results (either quick or long run), set setName="HighDim_paper" in the code, and also make sure the paths to the paper dataset are correctly set and you have the data.

To do your own test, copy the parameter setting (the code right after if (strcmp(testName, "NewTest_redosim")) and create your own named test with changed simulation parameters).

The debias function described in the paper is contained in the lkines right after the code heading:  "%%%%%%%%%%%%%%% Estimate the debias shift %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".

Files included:

debiassim_bigloop_split_v2025a2.m
LassoSolver.m
NetSolver_layers.m
NetSolver_layers_norm.m
get_fitting_coefficients.m
param_class.m

cvx module

------------------------
This was tested most recently on Matlab 2024a and CVX release 2.2 (2024).



