%
% Script file: ex1_regularPencil.m
%
% This is a demo script for example1. We consider a synthetic buckling 
% eigenvalue problem 
%     K = lam*KG, 
% where K is positive semi-definite, KG is indefinite, and K-lam*KG is 
% regular. We demonstrate the rapid growth of the Lanczos vectors v_j 
% studied by Meerbergen and Stewart. We show the boundedness of the norm
% ||v_j||_2 through the regularization of the inner product.
%
% We will plot the following:
%   1. the angles between the Lanczos vectors v_j and the nullspace N(K); 
%   2. the growth of the nullspace components of the Lanczos vectors v_j;
%   3. the growth of the Lanczos vectors v_j;
%   4. the accuracy of the computed eigenpairs.
%

clear all
close all
format compact
format short e

n  = 500;
m  = 1;
sigma = -0.6;
tol = 1e-10;
maxit = 40;

% construct the matrices K and KG
rng(123,'twister');
lam = [(1:n-m)'; zeros(m,1)];
phi = (-1).^(1:n)';

Lam = diag( lam );
Phi = diag( phi );
Q   = orth(randn(n));
K  = Q*Lam*Q';  K  = 0.5*(K+K');
KG = Q*Phi*Q';  KG = 0.5*(KG+KG');

% Z is the orthonormal basis of the nullspace N(K)
% (K,KG) is a regular pencil and ZC=[].
Z = Q(:,n-m+1:n);

nrmK = norm(K,1);  nrmKG = norm(KG,1);

% the LDL^T factorization
[L,D,P] = ldl(K-sigma*KG);
InvKsKG = @(X) P*(L'\(D\(L\(P'*X))));

%
%----- Lanczos on C with K-inner product -----%
%
Mfun = @(X) K*X;
OpC  = @(X) InvKsKG(K*X);
applyP  = @(X) OpC(X);
checkcvg = @(mu,res,errtol) true(size(mu));

x0 = ones(n,1); 
v  = applyP(x0);  % starting vector v
nev = maxit;

[mu,X,ncg,iter,V] ...
    = LanFRO(OpC,Mfun,v,tol,checkcvg,nev,maxit);

lamK = sigma*mu./(mu-1);
[lamK,pp] = sort(lamK,'descend');
mu = mu(pp);
X  = X(:,pp);

Res = K*X - (KG*X)*diag(lamK);
backErrK = sqrt(sum(Res.*Res)')./ ...
  ((nrmK + abs(lamK).*nrmKG).*sqrt(sum(X.*X)'));

ncK   = sqrt( sum((Z'*V).*(Z'*V),1) );
nrmvK = sqrt( sum(V.*V) );
cosK  = ncK./nrmvK;

%
%----- Lanczos on C with M-inner product -----%
%
KGZ  = KG*Z;
Mfun = @(X) K*X + KGZ*(KGZ'*X);
OpC  = @(X) InvKsKG(K*X);
applyP  = @(X) OpC(X);
findCvg = @(mu,res) true(size(mu));

x0 = ones(n,1); 
v  = applyP(x0);  % starting vector v_0
nev = maxit;

[mu,X,ncg,iter,V] ...
    = LanFRO(OpC,Mfun,v,tol,checkcvg,nev,maxit);

lamM = sigma*mu./(mu-1);
[lamM,pp] = sort(lamM,'descend');
mu = mu(pp);
X  = X(:,pp);

Res = K*X - (KG*X)*diag(lamM);
backErrM = sqrt(sum(Res.*Res)')./ ...
  ((nrmK + abs(lamM).*nrmKG).*sqrt(sum(X.*X)'));

ncM   = sqrt( sum((Z'*V).*(Z'*V),1) );
nrmvM = sqrt( sum(V.*V) );
cosM  = ncM./nrmvM;

%fprintf('%d\t%+.10f\t%.2e\n', ...
%        [(1:length(lamM)); lamM'; backErrM';]);

%
%--------------- Plot the results ---------------%
%---------- plot the angles between the Lanczos vectors and N(K) ----------%
figure(1);
h1 = semilogy(1:length(ncK),cosK,'rx','MarkerSize',10,'linewidth',2);  hold on;  
h2 = semilogy(1:length(ncM),cosM,'b+','MarkerSize',10,'linewidth',2);  

box on;
set(gca,'FontName','Times New Roman','FontWeight','bold','FontSize',24)
xlim([0 length(ncM)+1]);  ylim([2e-17 1e+5])
xlabel('j-th step'); 
ylabel('cos\theta_j');
set(gca,'xtick',[1 10 20 30 40]);
set(gca,'ytick',[1e-16 1e-12 1e-8 1e-4 1 1e+4]);
legend([h1(1),h2(1)], ...
       'K-inner product', ...
       'M-inner product', ...
       'Location','northeast','FontSize',16);
   
%---------- plot the growth of the nullspace components ----------%
figure(2);
h1 = semilogy(1:length(ncK),ncK,'rx','MarkerSize',10,'linewidth',2);  hold on;  
h2 = semilogy(1:length(ncM),ncM,'b+','MarkerSize',10,'linewidth',2);  

box on;
set(gca,'FontName','Times New Roman','FontWeight','bold','FontSize',24)
xlim([0 length(ncM)+1]);  ylim([2e-17 1e+13])
xlabel('j-th step'); 
ylabel('||P_{N(K)}v_j||_2');
set(gca,'xtick',[1 10 20 30 40]);
set(gca,'ytick',[1e-16 1e-12 1e-8 1e-4 1 1e+4 1e+8 1e+12]);
legend([h1(1),h2(1)], ...
       'K-inner product', ...
       'M-inner product', ...
       'Location','northeast','FontSize',16);
   
%---------- plot the growth of the Lanczos vectors ----------%
figure(3);
h1 = semilogy(1:length(ncK),nrmvK,'rx','MarkerSize',10,'linewidth',2);  hold on;  
h2 = semilogy(1:length(ncM),nrmvM,'b+','MarkerSize',10,'linewidth',2); 

box on;
set(gca,'FontName','Times New Roman','FontWeight','bold','FontSize',24)
xlim([0 length(ncM)+1]);  ylim([0.1 2e+9])
xlabel('j-th step'); 
ylabel('||v_j||_2');
set(gca,'xtick',[1 10 20 30 40]);
set(gca,'ytick',[1 1e+4 1e+8]);
legend([h1(1),h2(1)], ...
       'K-inner product', ...
       'M-inner product', ...
       'Location','northeast','FontSize',16);
   
%---------- accuracy of the computed eigenpairs ----------%
figure(4);
h1 = semilogy(lamK,backErrK,'rx','MarkerSize',10,'linewidth',2);  hold on;
h2 = semilogy(lamM,backErrM,'b+','MarkerSize',10,'linewidth',2);

box on;
set(gca,'FontName','Times New Roman','FontWeight','bold','FontSize',24)
xlim([-21 21]); ylim([2e-17 10])
xlabel('eigenvalue'); 
ylabel('relative residual norm');
set(gca,'ytick',[1e-16 1e-12 1e-8 1e-4 1]);
legend([h1(1),h2(1)], ...
       'K-inner product', ...
       'M-inner product', ...
       'Location','northeast','FontSize',16);


