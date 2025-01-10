%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Jan 9, 2025
%  Written by Michael Newey
%  michael.newey@ll.mit.edu
%  mknewey@gmail.com
%  MIT Lincoln Laboratory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


cvx_path = './cvx/';
addpath (genpath([cvx_path]));

%CVX:  https://github.com/cvxr/CVX/releases
%https://cvxr.com/cvx/download/

%Note that this reduces bias on average, but it dosn't do it for every data sample and for every simulation
%specification.  You may have to average over enough tests to see the effect.

clear

%Change these paths to choose a new file
basepathname = '';
simdata_pathname = [basepathname, 'paperdata/'];
params.do_nn = 1; %Setting this (overrode by some tests below) to 1 turns on the neural network.  Setting to 2 normalizes the neural network inputs, and is probably a better version, but was not used in paper plots.

quick_test = 1; %IMPORTANT: Setting this to 1 significantly limits the number of epochs the neural net trains for, this makes a good initial test.
%Set to zero if you want to run a longer test across multiple simulation parameters

testName = "NewTest_redosim" %Changes do_nn to 2, and changes some sim parameters
testName = "HighDim_paper" %This runs on the same data files as was reported in the simulation

if (contains(testName, "redo"))
    redo_sim = 1;
else
    redo_sim = 0;
end

if (redo_sim) %Sim parameters
       
    params.Lx = 10000;  %Number of training data
    params.Lv = 10000; %Number of validation data
    params.Lz = 10000;  %Number of testing data
    params.L0 = 1000000;  %Number of reference data

    if (strcmp(testName, "NewTest_redosim"))
       
        params.shift_x = 0;
        params.sig_x = 1;
        params.sig_z = 1;
        params.shift_z = 1.1;%0.5
        params.scale_factor = 1;
    
        simfunc = @simulate_lasso
        uparams.dist = 10;
        uparams.rat = 0.01;
        uparams.extra_sample = 3;    

        params.alph = [0,1,0.7,0.5, 0.3];
        %params.Beta = [0.5,0.2,0.1,0.05,0.02,0.01];    
        params.polyorder = 4;
        params.nz = 0.1;
        params.Ndim = 6;
        params.perc_keep_b =0.4;

        params.b_fact = 4.5;     
        params.do_nn = 2;
    end

else
    sim_paths = dir([simdata_pathname, '*simdata*.mat']);
end

%params.netsize = [8,8];
params.netsize = [32,32,32,32];%The network size (e.g. 4 middle layers each with 32 nodes)

params.reg = 0.0002; %The regularization amount
params.lambda_lasso = 1;

loop_params.n_epoch = [2,10,100,500];
loop_params.netsize_mult = 1;

rng('shuffle')
rng(101)
datenum=datevec(now);

NITER = 60; %#ok<NASGU>
if (redo_sim)
    NTEST = 30;
else
    NTEST = length(sim_paths); %#ok<NASGU>
end
tic
if (quick_test)
    NTEST = 10;
    NITER = 10;
    %NTEST = 100
    %NITER = 100;
    loop_params.n_epoch = [2];%This is what makes it fast.  Setting it to really low training epoch so it is easy to see the bias reduction effect.
    %loop_params.n_epoch = [2,10];  This is an array, so you can evaluate as a function of number of epoch.
    replace_params = 1;    
    if (strcmp(testName, "HighDim_paper"))
        NTEST = 27;
        %NITER = 10;
        NITER = 60;
    end
end

%%
for bigger_iter = 1:NTEST  %The first loop is over simulation configuration
    %%

    dat = [];
    dat_sim = [];
    IterTest = 1;

    if (redo_sim)
        b_in = setup_poly_sim(params);

        params.Beta = b_in.*params.alph.';
        params.alph = params.alph(1:params.polyorder);

        params.Beta = params.Beta(:,1:params.Ndim);


        [X0, y_x0] = simfunc(params.L0, params.sig_x, params.shift_x, params.alph, params.Beta, params.nz, uparams, params.scale_factor);
        [Z0, y_z0] = simfunc(params.L0, params.sig_z, params.shift_z, params.alph, params.Beta, params.nz, uparams, params.scale_factor);      
    else
        rsim = load([sim_paths(bigger_iter).folder, '/', sim_paths(bigger_iter).name]);
        if (isfield(rsim, 'rdatonly'))
            rsim = rsim.rdatonly;
            datsim = rsim.dat;
        else
            datsim = rsim.dat_sim;
        end
        X0 = datsim.X0;
        y_x0 = datsim.y_x0;
        Z0 = datsim.Z0;
        y_z0 = datsim.y_z0;
        ind_us = strfind(sim_paths(bigger_iter).name,'_');
        testName = sim_paths(bigger_iter).name(ind_us(2)+1:ind_us(3)-1);
    end

    params0 = params;

    if (redo_sim || replace_params)
        params_parser = param_class([], loop_params,[]);
    else
        params_parser = rsim.params_parser;
        
        if (~quick_test)
            NITER = size(rsim.dat.X, 3)/ params_parser.N;
        end
    end

    %rsim.params_parser
    for IterParam = 1:params_parser.N  %This loop is over parameter specification (e.g. N_epoch(IterParam) )
        
        %We get our parameters using param_parser.
        params = params_parser.add_params(params, IterParam);
        n_epoch = params.n_epoch;

        for Iter1 = 1:NITER%This loop is over training and testing sample (each iteration is a new sample)
        
            fprintf('Configuration iteration %02d / %02d\n', bigger_iter, NTEST)
            fprintf('Total elapsed time %s\n', datestr(toc/86400, 'HH:MM:SS'));

            if (redo_sim)
                %%%%%%%%%%%%%%%%%%%%% Simulate the data %%%%%%%%%%%%%%%%
                %training data
                [X, y_x] = simfunc(params.Lx, params.sig_x, params.shift_x, params.alph, params.Beta, params.nz, uparams, params.scale_factor);
                %validation data
                [V, y_v] = simfunc(params.Lv, params.sig_x, params.shift_x, params.alph, params.Beta, params.nz, uparams, params.scale_factor);
                %test data
                [Z, y_z] = simfunc(params.Lz, params.sig_z, params.shift_z, params.alph, params.Beta, params.nz, uparams, params.scale_factor);           
                
            else%Load the data from file instead        
                X = datsim.X(:,:,IterTest);
                V = datsim.V(:,:,IterTest);
                Z = datsim.Z(:,:,IterTest);
                y_x = datsim.y_x(:,IterTest);
                y_v = datsim.y_v(:,IterTest);
                y_z = datsim.y_z(:,IterTest);
                params.Lz = rsim.params.Lz;
                params.Lx = rsim.params.Lx;
                params.Lv = rsim.params.Lv;
            end
          
            %%%%%%%%%%%%%%%%%%%%% Fit the data with a function %%%%%%%%%%%%%%%%
            if (params.do_nn > 0)
                % %%%%%%%%%%%%%% Solve NN %%%%%%%%%%%%%%%%%%%
                %netf = NetSolver(X,V,Z, [1,1], n_epoch);
                %netf = NetSolver(X,V,Z, netsize, n_epoch);
                netf = NetSolver_layers(X,V,Z, params.netsize, n_epoch, params.reg);
                netf.solve(y_x);
                netf.evaluate_bias(y_x0, y_x0, y_z0);
                use_func = netf;
            elseif (params.do_nn == 2)
                % %%%%%%%%%%%%%% Solve NN %%%%%%%%%%%%%%%%%%%
                %This version normalizes the inputs and outputs to prevent overly large values.              
                netf = NetSolver_layers_norm(X,V,Z, params.netsize, n_epoch, params.reg);
                netf.solve(y_x);
                netf.evaluate_bias(y_x0, y_x0, y_z0);
                use_func = netf;              
            else
                % %%%%%%%%%%%%%% Or solve lasso:%%%%%%%%%%%%%
                Lass = LassoSolver(X,V,Z, 2);
                Lass.solve(y_x, params.lambda_lasso)
                Lass.evaluate_bias(y_x0, y_x0, y_z0, Beta_truth);
                use_func = Lass;
            end

            params.do_lasso_also = 0;%Do both!
            if (params.do_lasso_also > 0)
                Lass = LassoSolver(X,V,Z, 2);
                Lass.solve(y_x, params.lambda_lasso)
                Lass.evaluate_bias(y_x0, y_x0, y_z0, Beta_truth);
                use_func = Lass;
            end

            %%%%%%%%%%%%%%%% Estimate the debias shift %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %While we do include a form of cross-fitting in this simulation, the easiest place to see the equations written out is
            %probably in section 2.4 "Debiased estimation without cross-fitting via lasso"  The final form of the debias is
            %very similar but here x_t in the final equation is replaced with a new independent sample (the variable "V").
            

            Lass_bias = LassoSolver(X,V,Z, 2);  
            Bx = Lass_bias.Xm;Bz = Lass_bias.Zm;Bv = Lass_bias.Vm;

            %This run the minimization algorithm for determining rho (p in th code)
            params.lambda = 1/10000; %Hardcoded fitting lambda
            cvx_begin
            variable p(size(Bz,2))
            minimize( - sum(Bz*p)/params.Lz*params.Lx + 1/2*p.'*Bx'*Bx*p + params.lambda*norm(p,1)*params.Lx);
            cvx_end

            %This claculates the term to be subtracted from the data
            debias_est = -p.'*Bv.'*(y_v - use_func.yest_v)/params.Lv;
            %Note that this is the number to subtract from the Z data to debias it.  It is not yet debiased at this point.

            %%%%%%%%%%%%%%% save the data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ybias_z= use_func.ybias_z; ybias_x = use_func.ybias_x;ybias_v = use_func.ybias_v;%rms_x = use_func.rms_x;rms_z = use_func.rms_z;
            std_x = use_func.std_x;std_z = use_func.std_z;
            yest_x = use_func.yest_x; yest_v = use_func.yest_v; yest_z = use_func.yest_z;

            rms_x = rms(y_x - yest_x);
            rms_z = rms(y_z - yest_z);

            %This accumulates data into a structure in a convenient way in order to save or use it later
            dat = SaveThisData(dat, IterTest, debias_est, ybias_z, ybias_x, ...
            yest_x, yest_v, yest_z, IterParam);
            if (redo_sim)
                dat_sim = SaveThisData(dat_sim, IterTest, X, V, Z, y_x, y_v, y_z);
            end

            IterTest = IterTest+1;

        end
    end

    dat_sim.y_x0 = y_x0;
    dat_sim.X0 = X0;
    dat_sim.y_z0 = y_z0;
    dat_sim.Z0 = Z0;
    
    if (~redo_sim)
        dat.simdata_pathname = simdata_pathname;
        dat.simfile = [sim_paths(bigger_iter).folder, '/', sim_paths(bigger_iter).name];
    end
    
    %%
    datestr0 = sprintf('%d', datenum(1:5));
    pathout = [basepathname, 'data_study2024/', datestr0, '/'];
    %pathout = ['C:\Users\MI22952\Documents\wnewey_work\data_study2024/',datestr0, '/'];
    mkdir(pathout);
    a=datevec(now);
    postfx = sprintf('%d', a(1:5));
    postfx = sprintf('%03d', bigger_iter);

    if (redo_sim)
        pathout_sim = [basepathname, 'data_study2024/', datestr0, '_simdat/']; mkdir(pathout_sim);
        %Save the sim data to simdata_pathname
        outname = [pathout_sim, 'simdata_',char(testName), '_', postfx, '.mat'];
        save(outname, 'dat_sim', 'uparams',  'params', 'params_parser','NITER');
        dat.simdata_pathname = pathout_sim;
        dat.simfile = outname;
    end

    outname = [pathout, 'fitting_debias_exp_',char(testName), '_', postfx, '.mat'];
    %save(outname, 'dat',   'uparams', 'params', 'params_parser','NITER');
    save(outname, 'dat',  'params', 'params_parser','NITER');

    %Some caluclations and data saving for later evaluation.  For a multi-day run, you would want to do this analysis by
    %loading saved data from a file instead.
    NP = length(mean(dat.yest_z))/NITER;
    res_mean_z = (reshape(mean(dat_sim.y_z0) - mean(dat.yest_z), NITER, NP));
    res_mean_z_deb = (reshape(mean(dat_sim.y_z0) -  mean(dat.yest_z - dat.debias_est), NITER, NP));

    res_mean_z_sv(:,bigger_iter) = res_mean_z;
    res_mean_z_deb_sv(:,bigger_iter) = res_mean_z_deb;

    mean_bias_x =  mean(reshape(dat.ybias_x, NITER, NP),1);
    std_bias_x = std(reshape(dat.ybias_x, NITER, NP),1);

    mean_bias_z =  mean(reshape(dat.ybias_z, NITER, NP),1);
    std_bias_z = std(reshape(dat.ybias_z, NITER, NP),1);

    mean_bias_z_corr =  mean(reshape(dat.ybias_z - dat.debias_est, NITER,NP),1);
    std_bias_z_corr = std(reshape(dat.ybias_z - dat.debias_est, NITER, NP),1);

    mean_bias_x_sv(bigger_iter) = mean_bias_x;
    std_bias_x_sv(bigger_iter) = std_bias_x;
    mean_bias_z_sv(bigger_iter) = mean_bias_z;
    std_bias_z_sv(bigger_iter) = std_bias_z;
    mean_bias_z_corr_sv(bigger_iter) = mean_bias_z_corr;
    std_bias_z_corr_sv(bigger_iter) = std_bias_z_corr;


end

toc
%%

fprintf('Mean absolute value of bias %f \n', mean(abs(mean_bias_z_sv)))
fprintf('Mean absolute value of bias after bias reduction %f \n', mean(abs(mean_bias_z_corr_sv)))

%%

fprintf('Mean absolute residual %f \n' , mean(abs(res_mean_z_sv(:))));
fprintf('Mean absolute residual after bias reduction %f \n' , mean(abs(res_mean_z_deb_sv(:))));

%%

%Setup the simulation parameters/weigths
function b_in = setup_poly_sim(params)
    
    Nord = length(params.alph);
    
    b_in =  (rand(Nord,params.Ndim) -.5)*params.b_fact;%Paper was about 3.5.
    
    perc_keep = 0.40;
    perc_keep = params.perc_keep_b;
    inds_zero = (rand(size(b_in)) > perc_keep);
    b_in_sv = b_in(:,[1,end]);
    b_in(inds_zero) = 0.1*rand(size(b_in(inds_zero))).*b_in(inds_zero);
    
    b_in(:,1) = b_in_sv(:,1)*1.2;    
    b_in(:,end) = b_in_sv(:,end)*0.2;

end

%Accumulate arbitrarty data into a common structure for later use
function dat= SaveThisData(dat, ind, varargin)
%dat =0
disp(inputname(1));

if (isempty(dat))
    for I=1:length(varargin)
        dat.(inputname(I+2)) = [];
    end
end

for I=1:length(varargin)
    curdat = varargin{I};
    nd = sum(size(curdat) > 1);
    if (nd < 2)
        curdat = curdat(:);
        nd = 1;
    end
    dat.(inputname(I+2)) = cat(nd+1, dat.(inputname(I+2)), curdat);
end

end

%Lasso based simulation
function [X_out, y_out, norm_fact] = simulate_lasso(L, sig_x, shift_x, alph, Beta, nz, uparams, scale_factor)
if (nargin < 7)
    uparams.dist = 0;
    uparams.rat = 0;
end

X_out = [];
y_out = [];
loop_number = 1;
if (0 && uparams.extra_sample > 0)
    loop_number = uparams.extra_sample;
end

for I=1:loop_number
    Ndim = length(Beta);
    X = randn(round(L*(1-uparams.rat)), Ndim)*sig_x + shift_x;
    X2 = uparams.dist*(rand(L*uparams.rat, Ndim)-0.5)*sig_x ;%+ shift_x;

    X = [X; X2];

    XB = (X*Beta.');
    y = zeros(size(XB,1),1);
    Beta_out = [];

    norm_fact = 2/50;

    norm_fact = 1/(mean(alph(:,2))*uparams.dist^2/4)/scale_factor;

    if (size(Beta,1) > 1)

        for I=1:length(alph)
            y = y +(XB(:,I).^(I-1));
        end
    else
        for I=1:length(alph)
            y = y + alph(I)*(XB.^(I-1));
        end
    end

    y = norm_fact*y + nz*randn(size(XB,1),1);  
    X_out = X;
    y_out = y;
end

end


%Some other simulations to try
function [X, y] = simulate_flat_ext(L, sig_x, shift_x, alph, Beta, nz, uparams, scale_factor)

if nargin < 7
    uparams.dist = 0;
    uparams.rat = 0;
end

[X,y] = simulate_lasso(L, sig_x, shift_x, alph, Beta,nz, uparams, scale_factor);

y = abs(y) - 0.1;
inds1 =  vecnorm(X,2,2) >2;
y(inds1)  = y(inds1) - 2;

end

function [X, y] = simulate_nonpoly(L, sig_x, shift_x, alph, Beta, nz, uparams, scale_factor)

if nargin < 7
    uparams.dist = 0;
    uparams.rat = 0;
end

[X,y] = simulate_lasso(L, sig_x, shift_x, alph, Beta,nz, uparams, scale_factor);

y = abs(y) - 0.1;
inds1 =  vecnorm(X,2,2) >2;
y(inds1)  = y(inds1) - 2;

end

function [X, y] = simulate_nonpoly2(L, sig_x, shift_x, alph, Beta, nz, uparams, scale_factor)

if nargin < 7
    uparams.dist = 0;
    uparams.rat = 0;
end

[X,y0] = simulate_lasso(L, sig_x, shift_x,  [0,1,1,0.1], [1,0.5,0.2,0.2,0.1,0.1], 0, uparams);

freq = 0.5;
%y = sinc(freq*2*pi*y0) +nz*randn(size(y0));

y = exp(-abs(y0)) .* sin(freq*2*pi*y0) +nz*randn(size(y0));

%y = abs(y) - 0.1;
%inds1 =  vecnorm(X,2,2) >2;
%y(inds1)  = y(inds1) - 2;

end


function [X, y] = simulate_nonpoly3(L, sig_x, shift_x, alph, Beta, nz, uparams, scale_factor)

if nargin < 7
    uparams.dist = 0;
    uparams.rat = 0;
end

[X,y0, norm_factor] = simulate_lasso(L, sig_x, shift_x, alph, Beta, 0, uparams, scale_factor);

freq = 2;
freq = 0.1;
freq = 0.3;
%freq = uparams.freq;
y = sin(freq*2*pi*y0)+nz*randn(size(y0));

%y = sum(Beta.*sin(2*f.*X),2);
%y = sinc(y*50) - 0.1
%y = abs(y) - 0.1;
%inds1 =  vecnorm(X,2,2) >2;
%y(inds1)  = y(inds1) - 2;

end


