%Esempio: obscured matrix A:

clear all;
close all;
clc;

A=[1 -1 NaN 4 5 ; 2 NaN -11 10.2 20 ; -1 9 12 NaN NaN;-34 50 -2 -1 NaN; 0.4 1.39 NaN 9 90]
SYMM = toeplitz(randi([0 9],6,1)); 

flag=0;
while(flag==0)
    SIMULAZIONE_INIZIALE = input('Numero della simulazione iniziale? ');
    %Y = floor( X ) rounds each element of X to the nearest integer less than or equal to that element.
    if floor(SIMULAZIONE_INIZIALE) == SIMULAZIONE_INIZIALE
        if SIMULAZIONE_INIZIALE>0 
            flag=1;
        end
    end
end

flag=0;
while(flag==0)
    NUMERO_SIMULAZIONI = input('Numero di altre simulazioni (minore o uguale a 50)? ');
    if floor(NUMERO_SIMULAZIONI) == NUMERO_SIMULAZIONI
        if NUMERO_SIMULAZIONI>0 
            if NUMERO_SIMULAZIONI <=50
                flag=1;
            end
        end
    end
end

N_SIMULAZIONI=SIMULAZIONE_INIZIALE+NUMERO_SIMULAZIONI;
display(['Simulations from ' num2str(SIMULAZIONE_INIZIALE) ' to ' num2str(N_SIMULAZIONI)])





for h=1:size(A,1)
    for k=1:size(A,2)
        if isnan(A(h,k)) % se ci sono dei nan nella matrice, li poniamo = 0
            A(h,k)=0;
        end
    end
end


colNames = {'x0','x1','x2', 'x3', 'x4'};
A = array2table(A,'VariableNames',colNames)

FLAGS_COLONNE={'x2', 'x3'};
%se vogliamo più colonne random senza ripetizione:
%ncol = 2 %number of columns you want
%x = randperm(size(A,2),ncol); %questo comando prende 3 numeri a caso da 1
%a 5 (size di A).
%FLAGS_COLONNE = A(:,x);
colonne2 = colNames;
portion_missing=0.25;

%facciamo un'unica simmulazione e vediamo che esce:

INDICI_RIGHE_MISSING=zeros(size(A,1),1);
NUMERO_MISSING=ceil(portion_missing*size(A,1));
% p = randperm( n , k ) returns a row vector containing k 
%unique integers selected randomly from 1 to n .
PERMUTAZIONE=randperm(size(A,1),NUMERO_MISSING+1);

flag_temp=0;
for INDICE_TEMP=1:(size(PERMUTAZIONE,2))
    INDICI_RIGHE_MISSING(PERMUTAZIONE(INDICE_TEMP))=1;
end

B=ones(size(A));
for h=1:size(INDICI_RIGHE_MISSING,1)
    if INDICI_RIGHE_MISSING(h)==1
        for k=1:size(colonne2,2)
            for l=1:size(FLAGS_COLONNE,2)
                %strcmp( s1,s2 ) compares s1 and s2 and returns 1 ( true ) if the two are identical and 0 ( false ) otherwise
                %cioè faccio la sostituzione laddove ho le colonne x2 e x3
                %(le flagged). QUeto vale solo nel casso in cui non
                %dovesssimo scegliere colonne random!
                if strcmp(char(colonne2(k)),char(FLAGS_COLONNE(l)))
                    B(h,k)=0;
              
                end
            end
        end
    end
  
end



A_mat = table2array(A);

size_training=sum(sum(B>0));

M=15;
N = 50;

lambda_tol_vector= zeros(M,1);
for k=1:M
    lambda_tol_vector(k)=2^(k-1);
end

clear CompletedMat CompletedMatrix CompletedMat_corrected;
clear Diff_sq Diff_sq_corrected Diff_sq_initial RMSE_initial_known RMSE_final_known;

for k=1:M
    
    lambda_tol = lambda_tol_vector(k);
    tol = 1e-9;
    fprintf('Completion using nuclear norm regularization... \n');
    [CompletedMat,objective,flag] = matrix_completion_nuclear_GG(A.*B,B,N,lambda_tol,tol);
    if flag==1 %flag viene 1 nel caso in cui non si abbiano singulaar values.
        CompletedMat=zeros(size(A));
    end
    
    CompletedMatrix{k}=CompletedMat;
    CompletedMat_corrected=CompletedMat; 
    % in pratica è come CompleteedMat ma mette 0 laddove un termine di
    % CompleetedMat è <0
    
    for index1=1:size(CompletedMat_corrected,1)
        for index2=1:size(CompletedMat_corrected,2)
            if CompletedMat(index1,index2)<0
                CompletedMat_corrected(index1,index2)=0;
            elseif CompletedMat(index1,index2)>100
                CompletedMat_corrected(index1,index2)=100;
            end
        end
    end
    
    CompletedMatrix_corrected{k}=CompletedMat_corrected;
    
    Diff_sq{k} = abs(CompletedMat-A).^2;
    Diff_sq_corrected{k} = abs(CompletedMat_corrected-A).^2;
    Diff_sq_initial{k} = abs(A).^2;
    RMSE_initial_known(k)=0;
    RMSE_final_known(k)=sqrt(sum2(Diff_sq{k}.*B)/sum(B(:)));
    
end
