function [ W, obj ] = MRDM( X_train, Y_train, para )

[num_train, num_feature] = size(X_train); num_label = size(Y_train, 2);

% calculate graph Laplacian
options1 = [];
options1.NeighborMode = 'KNN';
options1.k = 0;
options1.WeightMode = 'HeatKernel';
options1.t = 1;
S = constructW(X_train,options1);
D = diag(sum(S,2));
L = D - S;

% options2 = [];
% options2.NeighborMode = 'KNN';
% options2.k = 0;
% options2.WeightMode = 'Cosine';
% S0 = constructW(Y_train',options2);
% D0 = diag(sum(S0,2));
% L0 = D0 - S0;

% Initialize U
U = eye(num_feature);

H = eye(num_train) - 1 / num_train * ones(num_train, 1) * ones(num_train, 1)';

iter = 1; objp = 1;

while 1
    
    % Update Z
    % eigen decomposition
    T = para.alpha * L - para.beta * H * (Y_train * Y_train') * H - X_train * ((X_train' * X_train + para.gamma * U) \ X_train') + eye(num_train);
    [tmp_P,tmp_lambda] = eig(T);
    tmp_P = real(tmp_P);
    tmp_lambda = real(diag(tmp_lambda));
    [lambda,order] = sort(tmp_lambda, 'ascend');
    Z = tmp_P(:,order);
    Z = Z(:,1:num_label);

    
    
    % Update W
    W = (X_train' * X_train + para.gamma * U) \ (X_train'* Z);
    
    % Update U
    u = 0.5./sqrt(sum(W.*W, 2) + eps);
    U = diag(u);
    
    % objective function    
    obj(iter) =  para.alpha * trace(Z'*L*Z) + (norm((X_train*W - Z), 'fro'))^2  - para.beta * trace(H*(Z*Z')*H*(Y_train*Y_train')) + para.gamma * sum(sqrt(sum(W.*W,2)+eps));

    cver = abs((obj(iter) - objp)/objp);
    objp = obj(iter);
    iter = iter + 1;

    % convergence condition
    if (cver < 10^-3 && iter > 20) , break, end
    
end

end

