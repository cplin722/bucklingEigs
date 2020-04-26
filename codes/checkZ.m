%
% Script file: checkZ.m
%
% We check the accuracy of the basis Z = [ZN ZC] 
% and the dimension of the common nullspace.
%

clear all
format compact
format short e
load ../Matrices/flighterJet_buckle.mat

nrigid = 6;
nrmK   = norm(K,1);  % 2.36e+10
nrmKG  = norm(KG,1); % 4.68e+06

Z  = [ZN ZC];
KZ = K*Z;
KGZ= KG*Z;

resKZ  = sqrt(sum(KZ.*KZ))./(nrmK*sqrt(sum(Z.*Z)));
resKGZ = sqrt(sum(KGZ.*KGZ))./(nrmKG*sqrt(sum(Z.*Z)));
kapZ   = cond(Z);

fprintf('The 1-norm of the matrix K is %.2e\n',nrmK);
fprintf('The 1-norm of the matrix KG is %.2e\n',nrmKG);
fprintf('Accuracy of the basis Z:');
fprintf('\ni\tresKZ\t\tresKGZ');
fprintf('\n%d\t%.2e\t%.2e',[(1:nrigid); resKZ; resKGZ]);
fprintf('\nThe conditioning of the basis Z: %.2e\n', kapZ);

% Check the dimension of the common nullspace.
[Y,~] = qr(Z,0);
KY  = K*Y;
KGY = KG*Y;
[~,D,~] = svd(KGY,0);

resKY = sqrt(sum(KY.*KY))./(nrmK*sqrt(sum(Y.*Y)));
sv    = diag(D);
omgY  = norm(Y'*Y-eye(nrigid),'fro');

fprintf('\nVerify the dimension of the common nullspace ...');
fprintf('\ni\tsv/nrmKG\tresKY');
fprintf('\n%d\t%.2e\t%.2e',[(1:nrigid); sv'/nrmKG; resKY;]);
fprintf('\nThe loss of orthogonality of the basis Y: %.2e\n', omgY);