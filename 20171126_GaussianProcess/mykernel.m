function kmn = mykernel(xm,xn,the)
sig=the;
kernel=@(x1,x2)sum(x1.*x2)/the;
[m,~]=size(xm);
[n,~]=size(xn);
K=zeros(m,n);
for i=1:m
  for j=1:n
    K(i,j)=kernel(xm(i,:),xn(j,:));
  end
end
kmn=K;
