function [U0, V0] = initialization_imc(L,X_left,X_right,p,r)
% Initialization phase for IMC_PF ---- compute the initial iteratre
%
%    Syntax
%
%       [U0, V0] = initialization_imc(L,F_left,F_right,p,r)
%
%    Description
%
%       initialization_imc takes,
%           L             - d-by-d observed data matrix
%           X_left        - d-by-n row feature matrix 
%           X_right       - d-by-n column feature matrix
%           p             - sampling rate
%           r             - rank
%       
%       returns
%           U0, V0        - initial iterate
%

% 1-step SVD
[A, Sigma, B] = svds(L/p,r);    
U0 = X_left.' * A * sqrt(Sigma);
V0 = X_right.' * B * sqrt(Sigma);

end

