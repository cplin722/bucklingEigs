function [mu,X,ncg,iter,varargout] = ...
    LanFRO(OpC,Mfun,v,checkcvg,nev,maxit)
% [mu,X,ncvg,iter,varargout]=LanFRO(OpC,Mfun,findCvg,v,checkcvg,nev,maxit)
% The function LanFRO returns the eigenpairs of the ordinary eigenproblem
%     C*x = mu*x,
% where the matrix C is symmetric respect to the M-inner product.
%
% Inputs:
%  * OpC (function_handle) - matrix-vector product u=Cv.
%  * Mfun (function_handle) - matrix-vector product u=Mv. 
%  * v (class numeric) - starting vector.
%  * checkcvg (function_handle) - checkcvg(mu,res) returns a mask 
%    identifying the converged eigenpairs. mu(mask) are the converged 
%    eigenvalues and S(:,mask) are the associated eigenvectors.
%  * nev (class numeric) - number of desired eigenpairs.
%  * maxit (class numeric) - maximal number of iterations.
%
% Outputs:
%  * mu (class numeric) - computed eigenvalues of C.
%  * X (class numeric) - the associated computed eigenvectors.
%  * ncg (class numeric) - number of converged eigenpairs.
%  * iter (class numeric) - number of iterations to reach convergence. 
%    iter = -1 if the maximal number of iterations reached.
%
% Date: 19-Apr-2020
%

n  = size(v,1);
vp = zeros(n,1);  % v_{j-1}
vc = zeros(n,1);  % v_{j}
a  = zeros(1,maxit);  % vector for the diagonal entries of T
b  = zeros(1,maxit);  % vector for the superdiagonal entries of T,
                      % the superdiagonal starts at b(1)
mu = [];
X  = [];
V  = [];

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
    
    %----- We perform full reorthogonalization -----%
    h = V'*p;
    r = r - V*h;
    %-----------------------------------------------%
    p = Mfun(r);
    a(j) = alph;  % update the diagonal of T_j
    if j > 1
        b(j-1) = beta;  % update the superdiagonal of T_j
    end
    beta = realsqrt( p'*r );
    %----- Compute the eigenvalue decomposition of T_j -----%
    if j == 1
        T = a(j);
    else
        T = diag(b(1:j-1),-1) + diag(a(1:j),0) + diag(b(1:j-1),1);
    end
    [S,mu] = eig(T,'vector');
    %----- Check convergence -----%
    res  = abs(beta*S(j,:))';
    mask = checkcvg(mu,res);  
    ncg  = sum(mask);
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