%% karmarkar's algorithm

function x = karmarkar(A,b,c,x)

N = size(c,1);
x0 = x;
y0 = ones(N,1);
epi = 1e-3;
alpha = 0.8;

x_c = 2*ones(N,1);
x_n = x0;

r = zeros(N,1);
Dx = diag(x_c);
flag = 0;
i = 1;
while flag == 0
    
    
    x_c = x_n;
    Dx = diag(x_c);
    
    r = c - A'*inv((A*Dx*Dx*A'))*A*Dx*Dx*c;
    p = -Dx*Dx*r;
    x_n = x_c + alpha*p/norm(Dx*r) ;
    
    if (r > 0)&(ones(1,N)*Dx*r <epi)
        flag =1;
    end
    i = i+1;
    x(:,i) = x_n;
end 
end