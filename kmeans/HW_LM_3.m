clear
load('mnist_all.mat')

data(1:5923, :) = train0;
data(5924:12665, :) = train1;
data(12666:18623, :) = train2;
data(18624:24754, :) = train3;
data(24755:30596, :) = train4;
data(30597:36017, :) = train5;
data(36018:41935, :) = train6;
data(41936:48200, :) = train7;
data(48201:54051, :) = train8;
data(54052:60000, :) = train9;


label(1:5923, 1) = 0;
label(5924:12665, 1) = 1;
label(12666:18623, 1) = 2;
label(18624:24754, 1) = 3;
label(24755:30596, 1) = 4;
label(30597:36017, 1) = 5;
label(36018:41935, 1) = 6;
label(41936:48200, 1) = 7;
label(48201:54051, 1) = 8;
label(54052:60000, 1) = 7;

data = double(data);
n = 60000;
d = 784;
k = 10;
l = 1:n;

% Initialize c1,...ck different centroids
rand_indices = [1, 5924, 12666, 18624, 24755, 30597, 36018, 41936, 48201, 54052];
c = data(rand_indices, :);
% c = randi([1, 500], [k, d]);

J = zeros(1, 8);
for b=1:8

    %  for each data point t set lt = min_j |xt − cj |^2
    for i= 1:n
        x  = data(i, :);
        l(i) = 1;
        dist  = (norm(x - c(1, :)))^2;
        for j= 1:k
            new_dist = (norm(x - c(j, :)))^2 ; 
            if (new_dist < dist)
                dist = new_dist;
                l(i) = j;
            end
        end
    end

    % For each j set: cj = mean of the current group.
    for j= 1:k
        S = zeros(1, d);
        H = 0;
        for i = 1:n
            x = data(i, :);
            if l(i) == j
                S = S + x;
                H = H + 1;
            end
        end
        c(j, :) = S/H;
    end

    % Calc the cost function.
    for i = 1:n
        x = data(i, :);
        dist = (norm(x - c(l(i), :)))^2;
        J(b) = J(b) + dist;
    end

end

plot(1:8, J)
hold on
xlabel('Iteration')
ylabel('Cost function')
title('K-Means algo for K = 10. Presenter: Ori Maimon')

R = 0;
for i=1:n
    if (label(i, 1) + 1) == l(i)
        R = R + 1;
    end
end

succesRate = R/n