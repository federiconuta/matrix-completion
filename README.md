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


