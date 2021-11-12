%%
% We need a functional environment so that the MAtimesVec subfunction can
% access variables in workspace
function [Anew,objective,flag] = matrix_completion_nuclear_GG(A,B,N,lambda_tol,tol)

%% Singular value thresholding for structured (sparse + low rank) matrix 

%clear;
%tol=10^-7;
%lambda_tol=0.5;

%%
% Read in sparse matrices downloaded from The University of Florida Sparse
% Matrix Collection
%data = load('mhd4800b.mat'); 
%A = data.mhd4800b;
%A=A(1:100,1:100);

%A=rand(100,5)*rand(5,100);
%[sA,vA,dA]=svd(A)

%%
% initialization
%B = rand(size(A))<0.3; % remove 10% of the entries

mat=A.*B;

m = size(mat,1);
n = size(mat,2);
L = zeros(m,1);  
R = zeros(n,1);

iteration=0;
criterion=0;
flag=0;
Anew=L*R';

while (criterion==0 && iteration<N && flag==0)

Aold=Anew;
    
iteration=iteration+1;
%iteration

%%
% Generation of structured matrix (sparse plus low rank)

%LR = L*R';                % generation of low rank matrix
%Amat = mat + LR;          % sparse + low rank (7.14) Tibshirani

%%
% Find all singular values >= lambda_tol by svt (deflation method). Function
% MAtimesVec is defined at end of this file.
%tic;
%svt is an algorithm to minimize the nuclear norm s.t. constraints. It
%computes the first m singular values;
%E.g. to request top 15 singular values and vectors, we use
%[U, S, V] = svt(A, 'k', 15)

[u,s,v] = svt(@MAtimesVec,'m',m,'n',n,'lambda',lambda_tol);
%toc;
%display(size(s));

if size(s,1)==0
    Anew=NaN*A;
    objective=NaN;
    flag=1;
    return
else
 
%size(s,1)
s=s-lambda_tol*eye(size(s,1));
L=u*s;
R=v;

    mat=(A-u*s*v').*B;
    Anew=L*R';
    objective(iteration)=1/2*(norm(mat,'Fro'))^2+lambda_tol*(sum(diag(s)));
    %objective(iteration)
    
    if (norm(Anew-Aold,'Fro')^2/(norm(Aold,'Fro')^2))<tol
        criterion=1;
    end
    
%norm(A-Anew,'Fro')    
end
end

if flag==1
    Anew=[];
    objective=[];
end

%%
% Subfunction for exploiting matrix structure of sparse plus low rank
function MAvec = MAtimesVec(vec, trans)

    if trans
       MAvec = (vec'*mat)' + R*(vec'*L)';
    else
       MAvec = mat*vec + L*(R'*vec);
    end
    
end


end

