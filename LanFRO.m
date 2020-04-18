function [mu,X,ncg,iter,varargout] = ...
    LanFRO(OpC,Mfun,findCvg,v,nev,maxit)
% [mu,X,ncvg,iter,varargout] = LanFRO(OpC,Mfun,findCvg,v,nev,maxit)
% The function LanFRO implements the Lanczos method on C with M-inner
% product. Full re-orthogonalization is used.
% The operator OpC is symmetric with respect to the Mfun-inner product. 
% The operator Mfun is positive semi-definite or positive definite. 
%
% INPUTS:
%   - OpC: function, Cx = \mu x is the targeted ordinary eigenproblem.
%   - Mfun: function, the M-inner product.
%   - findCvg: function, findCvg(mu,res) returns the mask for converged
%              eigenpairs.
%   - v: n-by-1 vector, the starting vector with n being the problem size.
%   - nev: integer, the number of eigenpairs we want to find.
%   - maxit: positive integer, the steps of Lanczos method to run.
%
% OUTPUTS:
%   - mu: ncvg-by-1 vector, the computed eigenvalues of OpC.
%   - X: n-by-ncvg matrx, each column is an associated computed 
%        eigenvector of OpC.
%   - ncg: integer, the number of converged eigenpairs.
%   - iter: integer, the number of steps LanFRO takes to converge,
%           iter = -1 when maximal number of iteration reached.
%
% Date of last update: 10/14/2019
%

n  = size(v,1);
vp = zeros(n,1);  % v_{j-1}
vc = zeros(n,1);  % v_{j}
a  = zeros(1,maxit);  % vector for the diagonal entries of T
b  = zeros(1,maxit);  % vector for the superdiagonal entries of T,
                      % the superdiagonal starts at b(2)
mu = [];
X  = [];
V  = [];
resHist = zeros(maxit,maxit);
muHist  = zeros(maxit,maxit);

r  = v;
p  = Mfun(r);
beta = realsqrt( p'*r );
for j = 1 : maxit
    vc = r / beta;
    V  = [V vc];
    r  = OpC(vc);  % v_{j+1}
    r  = r - beta*vp;
    
    p = Mfun(r);
    alph = vc'*p;
    r = r - alph*vc;
    p = Mfun(r);
    
    %----- Perform full reorthogonalization -----%
    h = V'*p;
    r = r - V*h;
    p = Mfun(r);
    
    a(j) = alph;  % update the diagonal of T_j
    if j > 1
        b(j) = beta;  % update the superdiagonal of T_j
    end
    beta = realsqrt( p'*r );
    %----- Compute the eigenvalue decomposition of T_j -----%
    if j == 1
        T = a(j);
    else
        T = diag(b(2:j),-1) + diag(a(1:j),0) + diag(b(2:j),1);
    end
    [S,mu] = eig(T,'vector');
    %----- Check convergence -----%
    res  = abs(beta*S(j,:))';
    mask = findCvg(mu,res);
    ncg  = sum(mask);  % number of converged eigenpairs
    if ncg >= nev  
        fprintf('LanFRO: convergence is reached at iteration %d.\n',j);
        iter = j;
        break;
    end
    %-------------------------------------%
    if j == maxit
        fprintf('LanFRO: the maximal number of iterations reached\n'); 
        iter = -1;
    end
    %-------------------------------------%
    vp = vc;
    vc = r; 
end

%----- Compute approximate eigenvectors of the converged eigenpairs -----%
ncg = sum(mask);
fprintf('LanFRO: %d eigenpairs computed!\n',ncg);
if ncg > 0
    mu = mu(mask);
    X  = V*S(:,mask);  
end

% additional outputs ...
if nargout > 0
    varargout{1} = V;
end

end