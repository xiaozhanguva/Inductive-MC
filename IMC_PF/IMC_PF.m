function [M_hat,dist,time] = IMC_PF(M_star,Omega,X_left,X_right,eta,p,max_iter,U0,V0)
%
%   Nonconvex gradient-based algorithm for inductive matrix compeltion
%
%    Syntax
%
%       [M_hat,dist,time] = IMC_PF(M_star,X_left,X_right,Omega,eta,p,total,U0,V0)
%
%    Description
%
%       IMC_PF takes:
%           M_star        - n-by-n target core low rank matrix 
%           Omega         - d-by-d observed index matrix
%           X_left        - d-by-n row feature matrix 
%           X_right       - d-by-n column feature matrix
%           eta           - the step size
%           max_iter      - maximum number of iterations
%           U0, V0        - initial iterate
%
%       returns:
%			M_hat         - recoverd low-rank matrix
%           dist          - the vector representing the relative distance 
%           time          - the vector representing the cpu time
%

L_star = X_left*M_star*X_right';    % the n-by-n unknown reconstructed low-rank matrix  
L_star_norm = norm(L_star,'fro');   % the Frobenius norm of L_star

% Phase1 -- initialization
U = U0; V = V0;
dist = zeros(max_iter,1);
time = zeros(max_iter,1);
cputime_init = cputime;

% Phase2 -- vanilla GD
for t = 1:max_iter    
    time(t) = cputime - cputime_init;
    mat_temp = X_left*U*V'*X_right'-L_star;    
    dist(t) = norm(mat_temp,'fro') / L_star_norm;   % compute the relative error
    %disp(dist(t))

    % calculate the gradient
    mat = mat_temp .* Omega;    
    nabla_U = 1/p*X_left'*mat*X_right*V + 0.5*U*(U'*U - V'*V);
    nabla_V = 1/p*X_right'*mat'*X_left*U + 0.5*V*(V'*V - U'*U);
  
    % gradient descent iterate
    U = U - eta * nabla_U; 
    V = V - eta * nabla_V; 
     
end
M_hat=U*V';

end
