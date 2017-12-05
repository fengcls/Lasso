% 0.5*||Ax - b||_2 + lambda*||x||_1
% Proximal Gradient Algorithm (PGA) 
function pga_lasso_func(A,b,lambda)
% step 1
L = 1;
ita = 1.1;
x(:,1) = zeros(size(A,2),1);
k = 1;
flag = 1;
f(k) = 0.5*norm(A*x(:,k)-b,2)^2+lambda*norm(x(:,k),1);

while (flag)
    tic
    % step 2
    y_o = x(:,k)-1/L*A'*(A*x(:,k)-b);
    
    % step 3
    z_o = sign(y_o).*max((abs(y_o)-lambda/L),0);
    
    % step 4
    if ((0.5*norm(A*z_o-b,2)^2)<=...
            ((0.5*norm(A*x(:,k)-b,2)^2)+...
            (A'*(A*x(:,k)-b))'*(z_o - x(:,k))+...
            L/2*norm(z_o-x(:,k),2)^2))
        
        x(:,k+1) = z_o;
        k = k+1;
        f(k) = 0.5*norm(A*x(:,k)-b,2)^2+lambda*norm(x(:,k),1);
        flag = (abs(f(k-1)-f(k))/f(k))>1e-5;
        
        t_lapse(k)=toc;
    else
        L = ita*L;
    end;
    
end;

figure;
plot(cumsum(t_lapse(11:end)),log(f(11:end)))
xlabel('cpu time (s)')
ylabel('log(prim obj)')
title('PGA')