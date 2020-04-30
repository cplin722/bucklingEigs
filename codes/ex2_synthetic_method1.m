%
% Script file: ex2_synthetic_method1.m
%
% This is a demo script for example2. We consider a synthetic buckling 
% eigenvalue problem 
%     K = lam*KG, 
% where K is positive semi-definite, KG is indefinite, and K and KG shares
% a common nullspace. We demonstrate the effectiveness of the shift-invert 
% Lanczos method. The matrix vector product is computed by solving the 
% augmented system
%     A[u; 0] = [K*v; 0]
% with A=[K-sig*KG ZC; ZC^T 0].
%
clear all
close all
format compact
format short e

n  = 2^10;
n2 = 4;
n3 = 2;
sig = -0.75;
intv  = [-1.5 0];
tol   = 1e-10;
maxit = 100;

% construct the matrices K and KG
lam = [10.^linspace(0,5,n-n2-n3)'; zeros(n2+n3,1)];
phi = [(-1).^(1:n-n3)'; zeros(n3,1)];

Lam = diag( lam );
Phi = diag( phi );
Q  = hadamard(n) / sqrt(n);  % Q is orthonormal
K  = Q*Lam*Q';   K  = 0.5*(K+K');
KG = Q*Phi*Q';   KG = 0.5*(KG+KG'); 

ZN = Q(:,n-n2-n3+1:n-n3);
ZC = Q(:,n-n3+1:n);

nrmK = norm(K,1);  nrmKG = norm(KG,1);
%
% ---------- Construct the operators ---------- %
%
A  = [K-sig*KG ZC; ZC' zeros(n3,n3)];  % the augmented matrix
[L,D,P_ldl] = ldl(A);
S  = eye(n+n3);

KGZN = KG*ZN;
nrm2KGZN = sum(KGZN.*KGZN)';
HN   = nrmK*diag(1./nrm2KGZN);
HC   = nrmK*eye(n3);

applyM  = @(X) K*X + KGZN*(HN*(KGZN'*X)) + ZC*(HC*(ZC'*X));
applyC  = @(X) OpC1(L,D,P_ldl,S,K,n,n3,X);

%
% ---------- Define the function to monitor convergence ---------- %
%
sig1 = intv(1);  sig2 = intv(2);
mu1 = sig1 / (sig1-sig);
mu2 = sig2 / (sig2-sig);
if sig < 0
    checkcvg = @(mu,res) ( (mu > mu1) | ((abs(mu) > tol) & (mu < mu2)) ) ...
                          & ((abs(sig)*res./((mu-1).^2))<tol);
else  % sig > 0
    checkcvg = @(mu,res) ( ((abs(mu) > tol) & (mu < mu1)) | (mu > mu2) ) ...
                          & ((abs(sig)*res./((mu-1).^2))<tol);
end

%
% ---------- Prior eigenvalue counting ---------- %
%
% compute inertia of ZN'*KG*ZN
[~,DInf,~] = ldl(ZN'*KG*ZN);
[npInf,nmInf] = inertiaOfD(DInf);

if sig1 == 0  % intv = (0 sig2)
    A2 = [K-sig2*KG ZC; ZC' zeros(n3,n3)];
    [~,D2,~] = ldl(A2);
    [np2,nm2] = inertiaOfD(D2);
    nev = nm2 - n3 - npInf;
elseif sig2 == 0  % intv = (sig1 0)
    A1 = [K-sig1*KG ZC; ZC' zeros(n3,n3)];
    [~,D1,~] = ldl(A1);
    [np1,nm1] = inertiaOfD(D1);
    nev = nm1 - n3 - nmInf;
else
    A1 = [K-sig1*KG ZC; ZC' zeros(n3,n3)];
    A2 = [K-sig2*KG ZC; ZC' zeros(n3,n3)];
    [~,D1,~] = ldl(A1);
    [~,D2,~] = ldl(A2);
    [np1,nm1] = inertiaOfD(D1);
    [np2,nm2] = inertiaOfD(D2);
    if (sig1 < 0) && (sig2 < 0)  % sig1 < sig2 < 0
        nev = nm1 - nm2;
    elseif (sig1 < 0) && (sig2 > 0)  % sig1 < 0 < sig2
        nev = nm1 + nm2 - 2*n3 - npInf - nmInf;
    else  % 0 < sig1 < sig2
        nev = nm2 - nm1;        
    end
end
if nev == 0
    error('No eigenvalues in the interval [%.1f, %.1f]!', sig1, sig2);
end

%
% ---------- Run the Lanczos method ---------- %
%
rng('default');
x0 = randn(n,1);  
v  = applyC(x0);

fprintf('\nRunning the shift-invert Lanczos method ...\n');
[mu,X,ncg,iter] = LanFRO(applyC,applyM,v,checkcvg,nev,maxit);
fprintf('\nThe shift-invert Lanczos method returns ...\n');

%
% ---------- Check the accuracy ---------- %
%
if ncg > 0
    lam = sig*mu./(mu-1);
    [~,pp] = sort(abs(lam),'ascend');
    lam = lam(pp);
    X   = X(:,pp);
    omgM = norm(X'*applyM(X)-eye(ncg),'fro');
    
    R  = K*X - (KG*X)*diag(lam);
    ZCtX = ZC'*X;
    
    relres = sqrt(sum(R.*R)')./ ...
        ( (nrmK+abs(lam)*nrmKG).*sqrt(sum(X.*X)') );
    costh = sqrt(sum(ZCtX.*ZCtX)')./sqrt(sum(X.*X)');
end

%
% ---------- Print results ---------- %
%
fprintf('\nPrint results ...\n');
fprintf('Matrix properties ...\n');
fprintf('1-norm of K:\t%.2e\n', nrmK);
fprintf('1-norm of KG:\t%.2e\n', nrmKG);
fprintf('Prior eigenvalue counting ...\n');
fprintf('Targeted interval: [%.1f, %.1f]\n', sig1, sig2);
if sig1 == 0
    fprintf('sig1 == 0; ');
else
    fprintf('nm1 = %d; ', nm1);
end
if sig2 == 0
    fprintf('sig2 == 0\n');
else
    fprintf('nm2 = %d\n', nm2);
end
fprintf('npInf = %d; nmInf = %d\n', npInf, nmInf);
fprintf('The number of the eigenvalues in the interval [%.1f, %.1f] is %d.\n', ...
            sig1, sig2, nev);
fprintf('Shift-invert Lanczos method ...\n');
fprintf('The shift is sig = %.2f\n', sig);
fprintf('# of non-zero elements of L is: %d\n', nnz(L));
fprintf('%d eigenpairs computed', ncg);
if ncg > 0  
    fprintf('\nk\tlam\t\trelres\t\tcos\n');
    fprintf('%2u\t%+.6e\t%.4e\t%.4e\n', ...
        [(1:ncg); lam';  relres'; costh']);
    fprintf('M-orthogonality of the computed eigenvectors: %.4e\n', omgM);
end