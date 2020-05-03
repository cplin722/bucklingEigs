%
% Script file: ex2_synthetic.m
%
% This is a demo script for example2. We consider a synthetic buckling 
% eigenvalue problem 
%     K = lam*KG, 
% where K is positive semi-definite, KG is indefinite, and K and KG shares
% a common nullspace. We demonstrate the effectiveness of the shift-invert 
% Lanczos method. The matrix-vector product u=Cv is computed by first 
% computing a solution u_p of the consistent singular linear system
%     (K - sig*KG)u_p = K*v.
% The vector u is computed by the projection
%     u = up - ZC*(ZC'*up),
% where the common nullspace basis ZC is orthonormal.
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

P = eye(n);
PtKsKGP = P'*(K-sig*KG)*P;  
[L,D,P_ldl] = ldl(PtKsKGP(1:n-n3,1:n-n3));
S = eye(n-n3);

KGZN = KG*ZN;
nrm2KGZN = sum(KGZN.*KGZN)';
HN   = nrmK*diag(1./nrm2KGZN);
HC   = nrmK*eye(n3);

applyM  = @(X) K*X + KGZN*(HN*(KGZN'*X)) + ZC*(HC*(ZC'*X));
applyC  = @(X) OpC(L,D,P_ldl,S,K,P,ZC,n,n3,X);

%
% ---------- Define the function to monitor convergence ---------- %
%
sig1 = intv(1);  sig2 = intv(2);
mu1 = sig1 / (sig1-sig);
mu2 = sig2 / (sig2-sig);
if sig < 0
    checkcvg = @(mu,res) ...
        ( (mu > mu1) | ((abs(mu) > tol) & (mu < mu2)) ) ...
           & ((abs(sig)*res./((mu-1).^2))<tol );
else  % sig > 0
    checkcvg = @(mu,res) ...
        ( ((abs(mu) > tol) & (mu < mu1)) | (mu > mu2) ) ...
          & ((abs(sig)*res./((mu-1).^2))<tol );
end

%
% ---------- Prior eigenvalue counting ---------- %
%
% compute inertia of ZN'*KG*ZN
[~,DInf,~] = ldl(ZN'*KG*ZN);
[npInf,nmInf] = inertiaOfD(DInf);

if sig1 == 0  % intv = (0 sig2)
    S2 = P'*(K-sig2*KG)*P;
    [~,D2,~] = ldl( S2(1:n-n3,1:n-n3) );
    [np2,nm2] = inertiaOfD(D2);
    nev = nm2 - npInf;
elseif sig2 == 0  % intv = (sig1 0)
    S1 = P'*(K-sig1*KG)*P;
    [~,D1,~] = ldl( S1(1:n-n3,1:n-n3) );
    [np1,nm1] = inertiaOfD(D1);
    nev = nm1 - nmInf;
else
    S1 = P'*(K-sig1*KG)*P;
    S2 = P'*(K-sig2*KG)*P;
    [~,D1,~] = ldl( S1(1:n-n3,1:n-n3) );
    [~,D2,~] = ldl( S2(1:n-n3,1:n-n3) );
    [np1,nm1] = inertiaOfD(D1);
    [np2,nm2] = inertiaOfD(D2);
    if (sig1 < 0) && (sig2 < 0)  % sig1 < sig2 < 0
        nev = nm1 - nm2;
    elseif (sig1 < 0) && (sig2 > 0)  % sig1 < 0 < sig2
        nev = nm1 + nm2 - npInf - nmInf;
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

fprintf('\nRunning the Lanczos method ...\n');
[mu,X,ncg,iter] = LanFRO(applyC,applyM,v,checkcvg,nev,maxit);
fprintf('\nThe Lanczos method returns ...\n');

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
fprintf('\nPrinting results ...\n');
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
fprintf('# of eigenpairs in the interval [%.1f, %.1f]: %d.\n', ...
            sig1, sig2, nev);
fprintf('\nResults of the shift-invert Lanczos method ...\n');
fprintf('The shift sig: %.2f\n', sig);
fprintf('# of non-zero elements of L: %d\n', nnz(L));
fprintf('# of eigenpairs computed: %d\n', ncg);

if ncg > 0  
    fprintf('Accuracy of the computed eigenpairs ...\n');
    fprintf('k\tlam\t\trelres\t\tcos\n');
    fprintf('%2u\t%+.6e\t%.4e\t%.4e\n', ...
        [(1:ncg); lam';  relres'; costh']);
    fprintf('M-orthogonality of the computed eigenvectors: %.4e\n', omgM);
end