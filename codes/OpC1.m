function Y = OpC1(L,D,P_ldl,S,K,n,n3,X)

nb = size(X,2);
B  = [K*X; zeros(n3,nb)];
Y  = S*(P_ldl*(L'\(D\(L\(P_ldl'*(S*B))))));
Y  = Y(1:n,:);