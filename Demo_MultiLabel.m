%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code is a demo of the Nonconvex gradient-based IMC algorithm on 
% multi-label data Yahoo Arts [1] when the percentage of observed 
% training instances' label assignments is 10%.
% 
% [1] N. Ueda, K. Saito. Parametric mixture models for multi-label text. 
% In: NIPS, 2002, 721-728.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all

%% Load the Arts data
addpath('IMC_PF');
dataname='Arts';
data=['data/' dataname '.mat'];
load(data);

p = 0.1;    % observed percentage
percent = 1-p;    
U=[train_data;test_data];
hatT=[train_target;test_target];
[n,q]=size(hatT);

%% Generate the observed data matrices
trial_train = 1;
s = RandStream.create('mrg32k3a','seed',trial_train);
RandStream.setGlobalStream(s);

% Generate Omega
train_num=round(n*p);

% Generate p(%) training and 1-p(%) testing
obrT=zeros(size(hatT));
indexperm=randperm(n);
train_index=indexperm(1,1:train_num);
test_index=indexperm(1,train_num+1:n);
remainT=hatT(train_index,:);

% Generate the percent% randomly observed entries in the training data
for iii=1:q
    positive_index=find(remainT(:,iii)>0);
    positive_number=length(positive_index);
    positive_random=randperm(positive_number);
    positive_select=positive_index(positive_random(1,1:ceil(positive_number*percent)),1);
    
    negative_index=find(remainT(:,iii)<=0);
    negative_number=length(negative_index);
    negative_random=randperm(negative_number);
    negative_select=negative_index(negative_random(1,1:ceil(negative_number*percent)),1);
    
    obrT(train_index(1,positive_select),iii)=1;
    obrT(train_index(1,negative_select),iii)=1;
end

L_Omega=hatT.*obrT;
Omega_linear=find(obrT);
X_right=eye(size(L_Omega,2));

if  min(min(hatT))==-1
    hatT=(hatT+1)/2;
end

%% perform gradient-based algorithm
% set the parameters
dim_feature_PF = 100;
rank = 20;
eta = 0.001;
maxiter = 100;

[X_left,~,~] = svds(U,dim_feature_PF);
[U0,V0] = initialization_imc(L_Omega,X_left,X_right,p,rank);
        
[M_hat,~] = IMC_PF_real(L_Omega,obrT,X_left,X_right,eta,p,maxiter,U0,V0); 
testPrecision_PF = PerformanceMeasure(X_left*M_hat*X_right',hatT,test_index);

disp(['The feature dimension is ' num2str(dim_feature_PF)]);
disp(['The rank of M_hat is ' num2str(rank)]);
disp(['The step size (eta) is ' num2str(eta)]);
disp(['The average precision for testing data is: ' num2str(testPrecision_PF)]);
