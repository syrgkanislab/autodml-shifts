%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Jan 9, 2025
%  Written by Michael Newey
%  michael.newey@ll.mit.edu
%  mknewey@gmail.com
%  MIT Lincoln Laboratory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [Xm, m_Xm, s_Xm] = get_fitting_coefficients(X, N, m_Xm, s_Xm)

    if (N > 2)
        disp('Need to make this handle higher order coefficients')
    end
    Xsq = [];
    for I=2:N%Make this higher order
        Xsq = reshape(X.*permute(X, [1,3,2]), size(X,1), size(X,2)^2);
        %Xm = [X, Xsq ]; 
    end
   
    Xm = [ones(size(X(:,1))), X, Xsq ];
    if (nargin < 3)
        s_Xm = std(Xm);m_Xm = mean(Xm);
    end
    if (0)
    Xm = (Xm - m_Xm) ./s_Xm;
    end
end
