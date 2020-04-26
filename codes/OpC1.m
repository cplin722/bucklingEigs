function Y = OpC1(L,D,P,S,K,n,n3,X)
% Y = OpC1(L,D,P_ldl,S,K,n,n3,X)
% Let ZC be a common nullspace basis of K and KG. Define the augmented 
% matrix A=[K-sig*KG ZC; ZC^T 0]. Let 
%     P'*S*A*S*P = L*D*L'
% be the factorization of A returned by the Matlab function ldl. The 
% function OpC1 returns the matrix-vector products Y=CX by solving the 
% augmented system
%     A[Y; 0] = [K*X; 0].
% 
% Inputs:
%  * L (class numeric) - triangular factor of the factorization.
%  * D (class numeric) - block diagonal of the factorization.
%  * P_ldl (class numeric) - permutation matrix of the factorization.
%  * S (class numeric) - scaling matrix of the factorization.
%  * K (class numeric) - stiffness matrix.
%  * n (class numeric) - size of the buckling eigenvalue problem.
%  * n3 (class numeric) - dimension of the common nullspace.
%  * X (class numeric) - the vectors we multiply C by.
%
% Outputs:
%  * Y (class numeric) - matrix-vector products Y=CX.
%
nb = size(X,2);
B  = [K*X; zeros(n3,nb)];
Y  = S*(P*(L'\(D\(L\(P'*(S*B))))));
Y  = Y(1:n,:);