%%
% We need a functional environment so that the MAtimesVec subfunction can
% access variables in workspace
function[] = script_test

%% Singular value thresholding for structured (sparse + low rank) matrix 

clear;
% Reset random seed
s = RandStream('mt19937ar','Seed',2014);
RandStream.setGlobalStream(s);

%%
% Read in sparse matrices downloaded from The University of Florida Sparse
% Matrix Collection
data = load('mhd4800b.mat'); 
A = data.mhd4800b;

%%
% initialization
B = rand(size(A))<0.90; % remove 10% of the entries

mat=A.*B;

m = size(mat,1);
n = size(mat,2);
L = zeros(m,1);  
R = zeros(n,1);

criterion=1;


while criterion==1
    
    
criterion=0;




%%
% Generation of structured matrix (sparse plus low rank)

LR = L*R';                % generation of low rank matrix
Amat = mat + LR;          % sparse + low rank

%%
% Find all singular values >= 0.2 by svt (deflation method). Function
% MAtimesVec is defined at end of this file.
tic;
[u,s,v] = svt(@MAtimesVec,'m',m,'n',n,'lambda',0.2);
toc;
display(size(s));

% %%
% % Find all singular values >= 0.2 by svt (succession method). Function
% % MAtimesVec is defined at end of this file.
% tic;
% [iu,is,iv] = svt(@MAtimesVec,'m',m,'n',n,'lambda',0.2,...
% 'method','succession');
% toc;
% display(size(is));
% 

%%
% Find all singular values >= 0.2 by full svd
FullAmat = full(Amat);
tic;
[su,ss,sv] = svd(Amat);
dss = diag(ss);
i = find(dss<=0.2);
su = su(:,1:i-1);
dss = dss(1:i-1);
sv = sv(:,1:i-1);
ss = diag(dss);
toc;
display(size(ss));

%Accuracy of solutions provided by svt deflation method
disp(norm(u*s*v'- su*ss*sv','fro')); 

% %%
% % Accuracy of solutions provided by svt succession method
% disp(norm(iu*is*iv'- su*ss*sv','fro')); 

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




