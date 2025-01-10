%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Jan 9, 2025
%  Written by Michael Newey
%  michael.newey@ll.mit.edu
%  mknewey@gmail.com
%  MIT Lincoln Laboratory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef NetSolver_layers < handle
     
    properties
        net; X;V;Z;yest_x; yest_v;yest_z;ybias_x;ybias_v;ybias_z;epochs;layers; options;
        std_x;std_v;std_z;
    end
    
    methods
        function obj = NetSolver_layers(X,V,Z, hiddenLayerSize, epoch, reg)

            if (nargin < 6)
                reg = 0;
            end

            layers = [  featureInputLayer(size(X,2)) ];
            for I=1:length(hiddenLayerSize)
                layers = [layers; fullyConnectedLayer(hiddenLayerSize(I), 'Name', ['fc', num2str(I)]); reluLayer];                                          
            end
                       
            miniBatchSize = size(X,1);
            miniBatchSize = size(X,1);
            sz_fact = 1;
            miniBatchSize = 1024*sz_fact;

            options = trainingOptions('adam', ...
                'InitialLearnRate', 0.01*sz_fact,...
                'MiniBatchSize',miniBatchSize, ...
                'Shuffle','every-epoch', ...
                'MaxEpochs',epoch,...%100
                'Verbose',true, ...                %'Verbose',false, ...
                'L2Regularization',reg);% ...
             
            obj.layers = layers;
            obj.options = options;
            
            obj.X = X/5;
            obj.V = V/5;
            obj.Z = Z/5;
            
        end
        
        function solve(obj, y_x)
            obj.layers = [obj.layers; fullyConnectedLayer(size(y_x,2)); regressionLayer];
            clear net0
            net0 = trainNetwork(obj.X,y_x,obj.layers,obj.options);
            
            obj.yest_x = predict(net0, obj.X);
            obj.yest_v = predict(net0, obj.V);
            obj.yest_z = predict(net0, obj.Z);
            obj.net = net0;       
        end
                
        function evaluate_bias(obj, ytrue, vtrue, ztrue)
            obj.ybias_x = mean(obj.yest_x) - mean(ytrue);
            obj.ybias_v = mean(obj.yest_v) - mean(vtrue);
            obj.ybias_z = mean(obj.yest_z) - mean(ztrue);
         
        end
        
    end
end


