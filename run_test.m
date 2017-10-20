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