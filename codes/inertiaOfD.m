function [np,nm] = inertiaOfD(D)
%
% This function counts the inertia of the symmetric, block diagonal D.
% The dimension of each block is 1 or 2.
%
% Inputs:
%   D: the n-by-n symmetric, block diagonal matrix.
%
% Outputs:
%   np: the number of positive eigenvalues of D.
%   nm: the number of negative eigenvalues of D.
%

n  = size(D,2);
np = 0;
nm = 0;

i = 1;
while i < n
   if D(i,i+1) == 0
       np = np + (D(i,i) > 0);
       nm = nm + (D(i,i) < 0);
       i  = i + 1; 
   else
       dd = eig(D(i:i+1,i:i+1));
       np = np + sum(dd > 0);
       nm = nm + sum(dd < 0);
       i  = i + 2;
   end
end

if i == n
   np = np + (D(n,n) > 0);
   nm = nm + (D(n,n) < 0);
end

end