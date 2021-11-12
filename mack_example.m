clear 
clc

A = [1 NaN; .3 4; NaN 6; 7 2];

B=1-isnan(A);

lambda_tol_vector= zeros(M,1);

conto = 1;
for h=-M:0.5:M
    lambda_tol_vector(conto)=2^(h);
    conto = conto+1;
end

for k=1:size(lambda_tol_vector)
    lambda_tol = lambda_tol_vector(k);
    tol = 1e-9;
    fprintf('Completion using nuclear norm regularization... \n');
    [CompletedMat,objective,flag] = matrix_completion_nuclear_GG(A.*B,B,N,lambda_tol,tol);
    if flag==1
        CompletedMat=zeros(size(A));
    end
    
    CompletedMatrix{k}=CompletedMat;
end
