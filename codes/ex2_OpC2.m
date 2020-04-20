%
% Script files: ex2_OpC2.m
%
% This is a demo script for example2.
% The matrix-vector product u=Cv is computed by Method 2.
%

clear all
format compact
format shortE
load ../Matrices/flighterJet_buckle.mat

n     = size(K,2);
sig   = 4.0;
intv  = [0.0 8.0];
tol   = 1e-6;
maxit = 200;

tau   = 0.1;
nrmK  = norm(K,1);  nrmKG = norm(KG,1);

%
% ---------- Construct the operators ---------- %
%
% Find the permutation matrix first. 
% Use greedy approach.
%
colnnz = sum((K-sig*KG)~=0);
[~,i1] = max(colnnz(1:6:n));
[~,i2] = max(colnnz(2:6:n));
[~,i3] = max(colnnz(3:6:n));
j1 = 1 + (i1-1)*6;
j2 = 2 + (i2-1)*6;
j3 = 3 + (i3-1)*6;

P = speye(n);
P(:,[j1 n-2]) = P(:,[n-2 j1]);
P(:,[j2 n-1]) = P(:,[n-1 j2]);
P(:,[j3 n])   = P(:,[n j3]);

t0_ldl=tic;
n3 = size(ZC,2);
PtKsKGP = P'*(K-sig*KG)*P;  
[L,D,P_ldl,S] = ldl(PtKsKGP(1:n-n3,1:n-n3),tau);
time_ldl=toc(t0_ldl);

KGZN = KG*ZN;
nrm2KGZN = sum(KGZN.*KGZN)';
HN   = nrmK*diag(1./nrm2KGZN);
HC   = nrmK*eye(3);

applyM  = @(X) K*X + KGZN*(HN*(KGZN'*X)) + ZC*(HC*(ZC'*X));
applyC  = @(X) OpC2(L,D,P_ldl,S,K,P,ZC,n,n3,X);

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
    [~,D2,~,~] = ldl( S2(1:n-n3,1:n-n3) );
    [np2,nm2] = inertiaOfD(D2);
    nev = nm2 - npInf;
elseif sig2 == 0  % intv = (sig1 0)
    S1 = P'*(K-sig1*KG)*P;
    [~,D1,~,~] = ldl( S1(1:n-n3,1:n-n3) );
    [np1,nm1] = inertiaOfD(D1);
    nev = nm1 - nmInf;
else
    S1 = P'*(K-sig1*KG)*P;
    S2 = P'*(K-sig2*KG)*P;
    [~,D1,~,~] = ldl( S1(1:n-n3,1:n-n3) );
    [~,D2,~,~] = ldl( S2(1:n-n3,1:n-n3) );
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
t0_lan=tic;
[mu,X,ncg,iter] = LanFRO(applyC,applyM,v,checkcvg,nev,maxit);
time_lan=toc(t0_lan);
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
fprintf('The shift sig: %.1f\n', sig);
fprintf('# of non-zero elements of L: %d\n', nnz(L));
fprintf('# of eigenpairs computed: %d\n', ncg);

if ncg > 0  
    fprintf('Accuracy of the computed eigenpairs ...\n');
    fprintf('k\tlam\t\trelres\t\tcos\n');
    fprintf('%2u\t%+.6e\t%.4e\t%.4e\n', ...
        [(1:ncg); lam';  relres'; costh']);
    fprintf('M-orthogonality of the computed eigenvectors: %.4e\n', omgM);
end
fprintf('\nTiming by tic-toc ...\n');
fprintf('ldl of the submatrix S11: %.3e secs\n', time_ldl);
fprintf('LanFRO: %.3e secs\n', time_lan);