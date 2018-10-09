function [M_hat, flag] = IMC_PF_real(L,Omega,X_left,X_right,eta,p,maxiter,U0,V0)
%
%   Nonconvex inductive matrix compeltion algorithm for multi-label learning
%
%    Syntax
%
%       [M_hat, flag] = IMC_PF_real(L,Omega,X_left,eta,p,maxiter,U0,V0)
%
%    Description
%
%       IMC_PF takes:
%           L             - d-by d observed low-rank matrix 
%           Omega         - d-by-d observed index matrix
%           X_left        - d-by-n row feature matrix 
%           X_right       - d-by-n column feature matrix (identity)
%           eta           - step size
%           maxiter       - maximum number of iterations
%           U0, V0        - initial iterate
%
%       returns:
%			M_hat         - recoverd low-rank matrix
%           flag          - flag = 1, if successful recovery; flag = 0,
%                           otherwise
%

tol = 1e-3;     % set the tolerance parameter

% Phase1 -- initialization
U = U0; V = V0;
dist = zeros(maxiter,1);
flag = 0;

% Phase2 -- gradient descent
for t = 1:maxiter  
    mat = (X_left*U*V'*X_right'-L) .* Omega;
    dist(t) = norm(mat,'fro')^2;
             
    nabla_U = 1/p*X_left'*mat*X_right*V + 0.5*U*(U'*U - V'*V);
    nabla_V = 1/p*X_right'*mat'*X_left*U + 0.5*V*(V'*V - U'*U);  
    U = U - eta * nabla_U; 
    V = V - eta * nabla_V;      
        
    if t>1 
        if dist(t) > dist (t-1)
            warning('step size too large, does not converge');
            break;
        elseif abs(dist(t)-dist(t-1))<tol*dist(t)
            disp(strcat('successful recovery, end at the ',31,num2str(t),'th iteration'));
            flag = 1;
            break;    
        end
    end
end
M_hat = U*V';

if t==maxiter
     warning('reach maximum iteration');
     flag = 1;
end

end
