# MDWSVM
A Matlab to conduct multi-class classification using MDWSVM model. 
We develop an angle-based multicategory distance-weighted support vector machine (MDWSVM) classification method that is motivated from the binary distance-weighted support vector machine (DWSVM) classification method. The new method has the merits of both support vector machine (SVM) and distance-weighted discrimination (DWD) but also alleviates both the data piling issue of SVM and the imbalanced data issue of DWD. Theoretical and numerical studies demonstrate the advantages of MDWSVM method over existing angle-based methods.
See the details of the method in https://pdfs.semanticscholar.org/a877/b837fbb0942dd3eccc3c1a00d3bc2af16e52.pdf

This subroutine is designed to calculate a linear discriminator into K classes based on the training data x_Train, the classification vector y_Train, and the violation cost penalty CC. 

The user will need to get the SDPT3 optimization package, Version 3.02, available at http://www.math.nus.edu.sg/~mattohkc/sdpt3.html, and install it. We recommend that sep.m be run from the SDPT3 directory.

SDPT3 includes an m-file startup.m, which sets the path, default parameters, and global environment variables. If the user has his/her own startup routine, this may not perform the necessary tasks. Either the instructions in SDPT3's startup.m should be appended, or SDPT3's startup.m be renamed startupSDPT3.m, say, and this called at the appropriate place.

% The subroutine contains various commented-out statements, which
% can be reinstated by those users wishing to see the time taken in
% various steps, the numbers of variables and constraints, etc.

% Input:  X, a d x n matrix, whose columns contain the training data; n x 1 vector y, with y_j the class of vector x_j; and scalar penalty, the cost for perturbing each residual.

% Output: W: a d x K matrix, and 
%         beta: a K x 1 vector, such that x is classified in class k if the kth component of the K-vector W'x + beta is largest.
%         resid: the values of (w_k'x + beta_k) - (w_j'x + beta_j)
%            for x a column of X with label k.
%         totalviolation: the total amount added to the residuals.
%         dualgap: the duality gap, a measure of the accuracy
%            of the solutions found.
%         flag: an indication of the success of the computation:
%             0, success;
%            -1, inaccurate solution;
%            -2, problem infeasible or unbounded.

Running Example:
%%%%%% test example for MDWSVM
%generate data with three groups;
n1=200;n2=50;n3=50;
y_Train=[(1).*ones(1,n1) (2).*ones(1,n2) (3).*ones(1,n3)]';
x_Train=randn((n1+n2+n3),dim)*0.5;
x_Train(1:(n1+n2+n3),1:2)=x_Train(1:(n1+n2+n3),1:2)*1.1;
x_Train(1:n1,1)=x_Train(1:n1,1)+1;
x_Train(1:n1,2)=x_Train(1:n1,2);
x_Train((n1+1):(n1+n2),1)=x_Train((n1+1):(n1+n2),1)-0.5;
x_Train((n1+1):(n1+n2),2)=x_Train((n1+1):(n1+n2),2)+sqrt(3)/2;
x_Train((n1+n2+1):(n1+n2+n3),1)=x_Train((n1+n2+1):(n1+n2+n3),1)-0.5;
x_Train((n1+n2+1):(n1+n2+n3),2)=x_Train((n1+n2+1):(n1+n2+n3),2)-sqrt(3)/2;

%%%%%%%%%% run MDWSVM with constraint parameter C=4
[wSV,bSV]=DWSVMkLZ(x_Train,y_Train,4);
%%%%% wSV is the B matrix from the model, bSV is the intercept.
