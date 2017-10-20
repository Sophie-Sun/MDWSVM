function [w, betaa, outstat] = DWSVMkLZ(x_Train,y_Train, CC)
%%%%%%%%%%%%%% MDWSVM function. Use CVS package, need to install CVX package first.
if nargin<2 ;
    error('At least two arguments are needed for mdwsvm.m') ;
end ;

[n d]=size(x_Train);
y_case=unique(y_Train);
k=length(y_case);
XI=XI_gen(k);
y_Matrix=zeros(n,k-1);
alpha=0.5;

if(k==2)
    for ii=1:n;
    y_Matrix(ii,:)=XI(y_Train(ii));
    end
else
for ii=1:n;
    y_Matrix(ii,:)=XI(:,y_Train(ii));
end
end

%%%%%%% if dimension is larger than observations, use QR decomposition. 
if (d>n),
    [Q,XX]=qr(x_Train',0);
    dnew=n;
    x_Train=XX';
else 
    dnew=d;
end;

cvx_begin quiet ;
     cvx_solver SDPT3 ;
%    cvx_solver SeDuMi ;
    variable w(dnew, k-1) ;
    variable betaa(k-1,1) ;
    variable beta0(k-1,1);
    variable rho(n, 1) ;
    variable sve(n, 1) ;
    variable xi(n, 1) ;
    variable eta(n, 1) ;
    minimize( alpha*(ones(1, n)*(rho+sve+eta))+(1-alpha)*(ones(1, n)*xi)) ;
    subject to
        x_Train*w.*y_Matrix*ones((k-1),1)+y_Matrix*beta0+eta-rho+sve == 0 ;
        x_Train*w.*y_Matrix*ones((k-1),1)+y_Matrix*betaa+xi >= 1.0 ;
        for (i=1:n) ;
            {reshape([1, sve(i)], 2, 1), rho(i)} == lorentz(2) ;
        end ;
         sum(w.*w*ones((k-1),1))<= CC;
        eta == nonnegative(n) ;
        xi  == nonnegative(n) ;
        rho == nonnegative(n) ;
cvx_end ;

if(d>n)
    w=Q*w;
end;

normvalue=sqrt(sum(w.*w*ones((k-1),1)));
if normvalue ~= 1
    betaa = betaa/normvalue ;
    beta0 = beta0/normvalue ;
    w = w/normvalue ;
end


outstat.cvx_status = cvx_status ;
outstat.cvx_precision = cvx_precision ;
outstat.CC=CC ;
outstat.alpha = alpha ;
outstat.cvx_optval = cvx_optval ;
outstat.beta0 = beta0 ;