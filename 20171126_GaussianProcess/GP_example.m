function GP_example(kernel,Number_of_lines)
t=0:0.01:10;
N=length(t);

C=zeros(N);
for i=1:N
  for j=1:N
    C(i,j)=kernel(t(i),t(j));
  end
end
C=C+eye(N)*1E-5;
[E,p] = chol(C);
if p==0
  for i=1:Number_of_lines
    x=randn(1,N);
    y=x*E;
    hold on
    plot(t,y)
  end
  hold off
else
  error('not a Positive-definite kernel')
end