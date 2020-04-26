function Y = OpC2(L,D,P,S,K,T,ZC,n,n3,X)
% Y = OpC2(L,D,P_ldl,S,K,P,ZC,n,n3,X)
% Let T be a permutation matrix such that the leading (n-n3)-by-(n-n3)
% submatrix S11 of T'*(K-sig*KG)*T is non-singular. Let 
%     P'*S*S11*S*P = L*D*L',
% be the factorization of S11 returned by the Matlab function ldl. Let ZC
% be an orthonormal basis of the common nullspace of K and KG. The function 
% OpC2 returns the matrix-vector product Y=CX by first computing a solution 
% Yp of the consistent singular linear system
%     (K - sig*KG)Yp = K*X.
% The vectors Y are computed by the projection
%     Y = Yp - ZC*(ZC'*Yp).
% 
% Inputs:
%  * L (class numeric) - triangular factor of the factorization.
%  * D (class numeric) - block diagonal of the factorization.
%  * P (class numeric) - permutation matrix of the factorization.
%  * S (class numeric) - scaling matrix of the factorization.
%  * K (class numeric) - stiffness matrix.
%  * T (class numeric) - permutation matrix. The leading (n-n3)-by-(n-n3) 
%    submatrix S11 of T'*(K-sig*KG)*T is non-singular.
%  * ZC (class numeric) - orthonormal basis of the common nullspace.
%  * n (class numeric) - size of the buckling eigenvalue problem.
%  * n3 (class numeric) - dimension of the common nullspace.
%  * X (class numeric) - the vectors we multiply C by.
%
% Outputs:
%  * Y (class numeric) - matrix-vector products Y=CX.
%
nb = size(X,2);
B  = K*X;
B  = T'*B;
Y1 = S*(P*(L'\(D\(L\(P'*(S*B(1:n-n3,:)))))));
Y  = [Y1; zeros(n3,nb)];
Yp = T*Y;  % Yp is a solution of the consistent singular linear system
           %     (K-sig*KG)*Yp = B.
Y  = Yp - ZC*(ZC'*Yp);
