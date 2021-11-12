# Matrix Completion for prediction

This repository contains the additional material for the article "Matrix Completion of World Trade" by F.Nutarelli, G. Gnecco and M. Riccaboni that you can find [here](https://arxiv.org/abs/2109.03930).

In the first Section of this repository we introduce a step-by-step guide for the reader that is new to machine learning to guide her/him in understading Matrix Completion; in the second Section we provide further details on the main algorithm used for prediction tasks at different stages and on their implementation in Matlab.

# Matrix Completion: a theoretical resume
This code provides an easy way to run robust matrix completion.
Given a subset of observed entries of a matrix <img src="https://render.githubusercontent.com/render/math?math={\bf A} \in \mathbb{R}^{C \times P}">, Matrix Completion (MC) works by finding a suitable low-rank approximation (say, with rank <img src="https://render.githubusercontent.com/render/math?math=R">) of <img src="https://render.githubusercontent.com/render/math?math={\bf A}"> , by assuming the following model:

<img src="https://render.githubusercontent.com/render/math?math={\bf A}= {\bf C} {\bf G}^\top + {\bf W}\,,"> 

where <img src="https://render.githubusercontent.com/render/math?math={\bf C} \in \mathbb{R}^{C \times R}">, <img src="https://render.githubusercontent.com/render/math?math={\bf G} \in \mathbb{R}^{P \times R}">, whereas <img src="https://render.githubusercontent.com/render/math?math={\bf W} \in \mathbb{R}^{C \times P}"> is a matrix of modeling errors. The rank-<img src="https://render.githubusercontent.com/render/math?math={\bf C} \in \mathbb{R}^{C \times R}"> approximating matrix <img src="https://render.githubusercontent.com/render/math?math={\bf C} {\bf G}^\top"> is found by solving a suitable optimization problem, following (Mazumder et al., 2010):

<img src="https://render.githubusercontent.com/render/math?math=\underset{{\bf Z \in \mathbb{R}^{C \times P}}}{\rm minimize}  \left(\frac{1}{2} \sum_{(c,p) \in \Omega^{\rm tr}} \left(A_{c,p}-Z_{c,p} \right)^2 + \lambda \|{\bf Z}\|_*\right) \,, (1)"> 

where <img src="https://render.githubusercontent.com/render/math?math=\Omega^{\rm tr}"> is a training subset of pairs of indices <img src="https://render.githubusercontent.com/render/math?math=(c,p)"> corresponding to positions of known entries of the partially observed matrix <img src="https://render.githubusercontent.com/render/math?math={\bf A} \in \mathbb{R}^{C \times P}">, <img src="https://render.githubusercontent.com/render/math?math={\bf Z} \in \mathbb{R}^{C \times P}"> is the completed matrix (to be optimized), <img src="https://render.githubusercontent.com/render/math?math=\lambda \geq 0">
is a regularization constant (chosen by a suitable validation method), and <img src="https://render.githubusercontent.com/render/math?math=\|{\bf Z}\|_*">
is the nuclear norm of the matrix <img src="https://render.githubusercontent.com/render/math?math={\bf Z}">, i.e., the sum of all its singular values.
  
## The algorithm
Eq. (1) can be re-written as 

<img src="https://render.githubusercontent.com/render/math?math=\underset{{\bf Z} \in \mathbb{R}^{C \times P}}{\rm minimize} \left(\frac{1}{2} \|{\bf P}_{\Omega^{\rm tr}}({\bf A})-{\bf P}_{\Omega^{\rm tr}}({\bf Z})\|_F^2 + \lambda \|{\bf Z}\|_*\right)\,, (2)"> 

where, for a matrix <img src="https://render.githubusercontent.com/render/math?math={\bf Y} \in \mathbb{R}^{C \times P}">, <img src="https://render.githubusercontent.com/render/math?math=(P_{\Omega^{\rm tr}}({\bf Y}))_{c,p}:= Y_{c,p}"> if <img src="https://render.githubusercontent.com/render/math?math=(c,p) \in \Omega^{\rm tr}">, otherwise it is equal to 0. Here, <img src="https://render.githubusercontent.com/render/math?math=P_{\Omega^{\rm tr}}({\bf Y})"> represents the projection of <img src="https://render.githubusercontent.com/render/math?math={\bf Y}"> onto the set of positions of observed entries of the matrix <img src="https://render.githubusercontent.com/render/math?math={\bf A}">, and <img src="https://render.githubusercontent.com/render/math?math=\|{\bf Y}\|_F"> denotes the Frobenius norm of <img src="https://render.githubusercontent.com/render/math?math={\bf Y}"> (i.e., the square root of the summation of squares of all its entries).

The MC optimization problem (2) can be solved by applying the Algorithm below, named Soft Impute in Mazumder et al. (2010) (compared to the original version, here we have included a maximal number of iterations <img src="https://render.githubusercontent.com/render/math?math=N^{\rm it}">, which can be helpful to reduce the computational effort when one has to run the algorithm multiple times, e.g., for several choices of the training set <img src="https://render.githubusercontent.com/render/math?math=\Omega^{\rm tr}"> and of the regularization constant <img src="https://render.githubusercontent.com/render/math?math=\lambda">):
 
<img width="748" alt="pseudocode" src="https://user-images.githubusercontent.com/51603270/141461287-4141b82b-3ef9-457c-acac-673234231406.png">

In the above Algorithm, for a matrix <img src="https://render.githubusercontent.com/render/math?math={\bf Y} \in \mathbb{R}^{C \times P}$, ${\bf P}_{\Omega^{\rm tr}}^{\perp}({\bf Y})"> represents the projection of <img src="https://render.githubusercontent.com/render/math?math={\bf Y}"> onto the complement of <img src="https://render.githubusercontent.com/render/math?math=\Omega^{\rm tr}">, whereas 

<img src="https://render.githubusercontent.com/render/math?math={\bf S}_\lambda({\bf Y}):= {\bf U} \Sigma_\lambda {\bf V}^\top,"> 

being

<img src="https://render.githubusercontent.com/render/math?math={\bf Y}={\bf U} \Sigma {\bf V}^\top"> 

(with <img src="https://render.githubusercontent.com/render/math?math=\bm{\Sigma}={\rm diag} [\sigma_1,\ldots,\sigma_R]">) the singular value decomposition of <img src="https://render.githubusercontent.com/render/math?math={\bf Y}">, and 

<img src="https://render.githubusercontent.com/render/math?math=\Sigma_\lambda:={\rm diag} [(\sigma_1-\lambda)_+,\ldots,(\sigma_R-\lambda)_+]"> 

with <img src="https://render.githubusercontent.com/render/math?math=t_+:=\max(t,0)">.

It is worth mentioning that a particularly efficient implementation of the operator <img src="https://render.githubusercontent.com/render/math?math={\bf S}_{\lambda}(\cdot)"> is possible (by means of the MATLAB function svt.m, see Li and Zhou, (2017).


# Usage Examples:

In the following mock example we will create a Toepliz square matrix, <img src="https://render.githubusercontent.com/render/math?math={\bf A}">, and try to predict its entries through the matrix completion algorithm.
The algorithm composes of a main function, `matrix_completion_nuclear` and a complementary function, `svt`, by Li and Zhou, (2017). `svt.m` should be in the same directory as `matrix_completion_nuclear.m`

```
clear 
clc

%make sure you run this within the directory where svt.m is
cd /Users/federiconutarelli/Desktop/github_me/

A = toeplitz(randi([0 9],6,1));
B=1-isnan(A);

N = 10; %setting thee numbeer of iterations for matrix completion code
M=15; %choosing the ranges of lambda
lambda_tol_vector= zeros(M,1);

counter = 1;
for h=-M:0.5:M
    lambda_tol_vector(counter)=2^(h);
    counter = counter+1;
end

for k=1:size(lambda_tol_vector)
    lambda_tol = lambda_tol_vector(k);
    tol = 1e-9;
    fprintf('Completion using nuclear norm regularization... \n');
    [CompletedMat,objective,flag] = matrix_completion_nuclear(A.*B,B,N,lambda_tol,tol);
    if flag==1
        CompletedMat=zeros(size(A));
    end
    
    CompletedMatrix{k}=CompletedMat;
end
```

The results of the mock example are stored in `CompletedMatrix{k}`. In parrticular, a value for each regularization parameter <img src="https://render.githubusercontent.com/render/math?math=\lambda"> is provided as output. The "optimal" predicted matrix <img src="https://render.githubusercontent.com/render/math?math=\hat{A}"> is the one associated with the "optimal" <img src="https://render.githubusercontent.com/render/math?math=\lambda"> parameter. A simple elbow method can be adopted to select the latter.

# Citation
If you use this MC algorithm in your research, please cite us as follows:

F.Nutarelli, G.Gnecco. MC algorithm: A matlab algorithm for easy Matrix Completion.

BibTex
```
@misc{econml,
  author={F.Nutarelli, G.Gnecco},
  title={{MC algorithm}: {A matlab algorithm for easy Matrix Completion.}},
  howpublished={https://github.com/feedericonutarelli},
  year={2021}
}
```

# References

Li, C., & Zhou, H. (2017). **Svt:  Singular value thresholding in MATLAB.** Journal of Statistical Software, 81(2), DOI:10.18637/jss.v081.c02.

Mazumder, R., Hastie, T., & Tibshirani, R. (2010). **Spectral regularization algorithms for learning large incomplete matrices.** Journal of Machine Learning Research, 11, pp. 2287–2322.

Tibshirani, R. (1996). **Regression shrinkage and selection via the Lasso.** Journal of the Royal Statistical Society. Series B(Methodological), 58(1), pp. 267—288.

