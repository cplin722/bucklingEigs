function Y = OpC2(L,D,P_ldl,S,K,P,ZC,n,n3,X)

nb = size(X,2);
B  = K*X;
B  = P'*B;
Y1 = S*(P_ldl*(L'\(D\(L\(P_ldl'*(S*B(1:n-n3,:)))))));
Y  = [Y1; zeros(n3,nb)];
Yp = P*Y; 
Y  = Yp - ZC*(ZC'*Yp);
