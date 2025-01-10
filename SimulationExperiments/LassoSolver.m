%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Jan 9, 2025
%  Written by Michael Newey
%  michael.newey@ll.mit.edu
%  mknewey@gmail.com
%  MIT Lincoln Laboratory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef LassoSolver < handle
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        X;
        Z;
        V;
        Xm;
        Zm;
        Vm;
        yest_x;
        yest_v;
        yest_z;
        ybias_x;
        ybias_v;
        ybias_z;
        Beta_diff;
        Beta_L;
        lambda1
        
        
    end
    
    methods
        function obj = LassoSolver(X, V, Z, N)
            if (nargin < 4)
                N=2;
            end
            obj.X = X; obj.Z = Z; obj.V = V;
           
            [Xm, m_Xm, s_Xm] = get_fitting_coefficients(obj.X, N);
            Vm = get_fitting_coefficients(obj.V, N, m_Xm, s_Xm);
            Zm = get_fitting_coefficients(obj.Z, N, m_Xm, s_Xm);
            obj.Xm = Xm;  obj.Vm = Vm; obj.Zm = Zm;
        end        
                
        function solve(obj, y_x, lambda1)

            [Beta_Est_temp, fitinfo] = lasso(obj.Xm,y_x, 'Lambda', lambda1 );  
            obj.Beta_L = Beta_Est_temp;
            
            obj.yest_x = obj.Xm*obj.Beta_L;
            obj.yest_v = obj.Vm*obj.Beta_L;
            obj.yest_z = obj.Zm*obj.Beta_L;
            obj.lambda1 = lambda1;
        end
        
        function evaluate_bias(obj, ytrue, vtrue, ztrue, Btrue)
            obj.ybias_x = mean(obj.yest_x) - mean(ytrue);
            obj.ybias_v = mean(obj.yest_v) - mean(vtrue);
            obj.ybias_z = mean(obj.yest_z) - mean(ztrue);
        
            obj.Beta_diff = obj.Beta_L - Btrue;
        end
        
    end
end

