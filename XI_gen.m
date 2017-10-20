function XI=XI_gen(k)
%%%%%%%%%%%% calculate W vertex.
if (k==1)
    XI=[1 -1];
else
XI=zeros(k-1,k);
XI(:,1)=(k-1)^(-0.5)*ones(k-1,1);
for ii=2:k;
    XI(:,ii)=-(1.0+sqrt(k))/((k-1)^(1.5))*ones(k-1,1);
    XI(ii-1,ii)=XI(ii-1,ii)+sqrt(k/(k-1));
end
end
