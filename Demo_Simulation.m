%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is a demo for gradient based inductive matrix completion 
% algorithm on synthetic data with ambient dimension d=1000, feature 
% dimension n=100, rank r=10 and sampling rate p=0.1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all

%% Load the parameters and data matries
addpath('IMC_PF');

% set the matrix parameters
d = 1000;       % dimension of the observed matrix L
n = 100;        % dimension of the core low-rank matrix M_star
r = 10;         % rank of the unknown matrix M_star
p = 0.1;        % sampling probability of observed indices

% load the data matrices
filename = ['./Data/data_d_' num2str(d) '_n_' num2str(n), '_r_' num2str(r), '_p_' num2str(p) '.mat'];
load(filename);

%% Inductive matrix completion (gradient descent)
% initialization
L = (F_left*M_star*F_right') .* Omega;      % set the observed data matrix
[U0, V0] = initialization_imc(L,F_left,F_right,p,r);
eta = 0.25;
maxiter = 100;
[M_hat,dist,cputime] = IMC_PF(M_star,Omega,F_left,F_right,eta,p,maxiter,U0,V0);

% plot the iteration versus distance
plot(1:maxiter,dist,'LineWidth',1.5);
xlabel('Number of iterations','FontSize', 12);
ylabel('Relative error','FontSize', 12);

