% 0.5*||Ax - b||_2 + lambda*||x||_1
% subgradient method
function subgrad_func(A,b,lambda)
[~,n2] = size(A);
x = zeros(n2,1);
k=1;
g = ones(n2,1);
t = 0.0001;

t_lapse(1)=0;

while k<3 || (abs(f(k-1)-f(k-2))/f(k-1))>1e-5
    tic
    % f(round(k/10)+1)=0.5*norm(A*x-b,2)^2+lambda*norm(x,1);
    f(k)=0.5*norm(A*x-b,2)^2+lambda*norm(x,1);
    % the subgradient is A'*(A*x-b)
    s = x;
    s(x>0)=1;
    s(x<0)=-1;
    s(x==0) = -2*rand(length(find(x==0)),1)+1;
    g = A'*(A*x-b)+lambda*s;
    x = x - t*g;
    t_lapse(k)=toc;
    k = k+1;
end;
figure;
plot(cumsum(t_lapse(30:end)),log(f(30:end)))
xlabel('cpu time (s)')
ylabel('log(prim obj)')
title('Sub-grad')

