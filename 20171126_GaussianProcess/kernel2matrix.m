function [ kernel_matrix ] = kernel2matrix( design_matrix,kernel )
phi=design_matrix;
[N,~]=size(phi);
K=zeros(N,N);
for i=1:N
  for j=1:N
    K(i,j)=kernel(phi(i,:),phi(j,:));
%     K(j,i)=K(i,j);
  end
end

kernel_matrix=K;
end

