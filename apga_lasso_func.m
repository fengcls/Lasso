% 0.5*||Ax - b||_2 + lambda*||x||_1
% Accelerated Proximal Gradient Algorithm (APGA)
function apga_lasso_func(A,b,lambda)
% initialization
L = 1;
ita = 1.1;
x(:,1) = zeros(size(A,2),1);
w(:,1) = zeros(size(A,2),1);
k = 1;
theta(1) = 1;

f = [];
while k<3 || (abs(f(k-1)-f(k-2))/f(k-1))>1e-5
    f(k) = 0.5*norm(A*x(:,k)-b,2)^2+lambda*norm(x(:,k),1);
    tic
    % gradient step
    y_o = w(:,k)-1/L*A'*(A*w(:,k)-b);
    
    % proximal step
    z_o = sign(y_o).*max((abs(y_o)-lambda/L),0);
    
    % back-tracing step
    if ((0.5*norm(A*z_o-b,2)^2)<=...
            ((0.5*norm(A*w(:,k)-b,2)^2)+...
            (A'*(A*w(:,k)-b))'*(z_o - w(:,k))+...
            L/2*norm(z_o-w(:,k),2)^2))
        
        x(:,k+1) = z_o;
        theta(k+1) = 0.5+0.5*sqrt(1+4*theta(k)^2);
        w(:,k+1)=x(:,k+1)+(theta(k)-1)/(theta(k)+1)*(x(:,k+1)-x(:,k));
        
        t_lapse(k)=toc;
        k = k+1;
    else
        L = ita*L;
    end;
    
end;

figure;
plot(cumsum(t_lapse(11:end)),log(f(11:end)))
xlabel('cpu time (s)')
ylabel('log(prim obj)')
title('APGA')