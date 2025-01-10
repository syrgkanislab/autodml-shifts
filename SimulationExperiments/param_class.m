%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Jan 9, 2025
%  Written by Michael Newey
%  michael.newey@ll.mit.edu
%  mknewey@gmail.com
%  MIT Lincoln Laboratory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef param_class < handle
    %This handles parameter set, allowing you to specify parameter lists and the way of getting those parameters. 
    % (e.g. random -> monte carlo parameters)
    
    properties
        params = [];
        monte_carlo_params = [];
        N = 0;
        curiter = 0;
    end
    
    methods
        function obj = param_class(paired_params, gridded_params, monte_carlo_params)
            obj.monte_carlo_params = monte_carlo_params;
                        
            obj.params = obj.parse_gridded_params(obj.params, gridded_params);
            obj.params = obj.parse_paired_params(obj.params, paired_params);
            
            if (isempty(obj.params))
                return;
            end
            f = fieldnames(obj.params);                                    
            obj.N = length(obj.params.(f{1}));               
        end
        
        function params = add_params(obj, params, I)
            
            if (nargin > 2)
                params2 = obj.get_params(I);
            else
                params2 = obj.get_params();
            end
            
            f2 = fieldnames(params2);
            for I=1:length(f2)
                params.(f2{I}) = params2.(f2{I});
            end
        end
        
        function outp = get_params(obj, I)
           
            if (nargin < 2)
                obj.curiter =mod(obj.curiter, obj.N)+1;
            else
                obj.curiter = I;
            end
                                    
            if (~isempty(obj. monte_carlo_params))
                f = fieldnames(obj.monte_carlo_params)
                for I=1:length(f)
                    curp = obj.monte_carlo_params.(f{I});
                    outp.(f{I}) = curp(randi(length(curp)));
                end
            end
                        
            if (~isempty(obj.params))
                f = fieldnames(obj.params)
                for I=1:length(f)
                    outp.(f{I}) = obj.params.(f{I})(obj.curiter);
                end
            end
                            
        end
        
        function params = parse_gridded_params(obj, params, gridded_params)
            
            f = fieldnames(gridded_params);
            for I=1:length(f)                     
                cur_tune = gridded_params.(f{I});

                SZ(I) = length(cur_tune);        
            end

            for I=1:length(f)          
                cur_tune = gridded_params.(f{I});
                cur_ind_arr = [ones(1,I-1), length(cur_tune),1];

                dat = reshape(cur_tune, cur_ind_arr);

                SZ2 = SZ;
                SZ2(I) = 1;
                dat_out = repmat(dat, SZ2);
                
                paired_params.(f{I}) = dat_out(:);
            end            
            params = obj.parse_paired_params(params, paired_params);
        end
        
        function [params] = parse_paired_params(obj, params, paired_params)
            
            if (isempty(paired_params))
                return;
            end
                            
            if (isempty(params))
                params = paired_params;
                return;
            end
            
            f0 = fieldnames(paired_params);
            
            if (length(f0) == 0)
                return;
            end
            
            Ncur = length(paired_params.(f0{1}));%This better be the same for all or badness will hapen  %Put a check somewhere
                   
            
            f = fieldnames(params);
            N = length(params.(f{1}));
            for IterName = 1:length(f)
                curvals = params.(f{IterName});
                temp = repmat(curvals(:),1,Ncur);
                params.(f{IterName}) = temp(:);
            end
                
            f = fieldnames(paired_params);            
            for IterName = 1:length(f)
                curvals = paired_params.(f{IterName});            
                temp = repmat(curvals(:).',N,1);            
                params.(f{IterName}) = temp(:);
            end
            
            
        end
        
    end
end
