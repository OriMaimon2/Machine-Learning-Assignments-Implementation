clear
load('facesData.mat')
n = 1024;
step = 50;

A = zeros(n);
trainFaces = zeros(15*8, 1024);
testFaces = zeros(15*3, 1024);
trainIndexes = zeros(1, 15*8);
realTestIndexes = zeros(1, 15*3);

% Running on the persons:
for i = 1:15
    trainFaces(8*i-7:8*i, :) = faces(11*i-10:11*i-3, :); % Pictures for training.
    trainIndexes(1, 8*i-7:8*i) = i;
    testFaces(3*i-2:3*i, :) = faces(11*i-2:11*i, :); % Pictures for testing.
    realTestIndexes(1, 3*i-2:3*i) = i;
end

% PCA on the trainFaces:
% The mean face:
meanFace = mean(trainFaces); 

% Plots the mean face:
figure('Visible','off');
face = reshape(meanFace, [32,32]);
imagesc(face);
axis image;
hold on;
title('The mean face:');
hold off;
saveas(gcf, 'meanFace.png');
close(gcf);

% Finds the matrix A:
for j=1:(15*8)
        x = trainFaces(j, :);
        A = A + (x - meanFace)*((x - meanFace)');
end
A = A./n;
[V, D] = eig(A);

% Ploting the top 20 eigenfaces:
R = V(:, 1025-20:1024); % Saving only the first 20 eigen vectors.
R = R(:, end:-1:1);
figure('Visible','off');
title("Top 20 eigen faces")
for j=1:20
    subplot(3, 7, j);
    eigFace = R(:, j);
    eigFace = reshape(eigFace', [32,32]);
    imagesc(eigFace);
    axis image;
    hold on;
    title(sprintf('V %d', j));
    hold off;
end
saveas(gcf, 'eigFace.png');
close(gcf);

d = 0;
Acc = zeros(1, floor(1024/step + 1));
for K=1:step:1024
    P = V(:, 1025-K:1024); % Saving only the first K eigen vectors.
    P = P(:, end:-1:1);

    % Extract top K components of the train faces:
    trainPCA = zeros(15*8, K);
    for l=1:15*8
        x = trainFaces(l, :);
        trainPCA(l, :) = x*P;
    end

    % Comparison between pca function and our PCA code:
    % newPCA = pca(trainFaces', 'NumComponents', K);
    % norm(newPCA - PCA)

    % Compute its K-dimensional projection of the test images onto the face
    % space:
    testPCA = zeros(15*3, K);
    for l=1:15*3
        x = testFaces(l, :);
        testPCA(l, :) = x*P;
    end

    % For each test image, find the training image that is "closest” (in
    % the sense of Euclidean distance) to the test image in the face space, and assign
    % the label (person index) of the training image to the test image:
    testIndexes = zeros(1, 15*3);
    for a=1:15*3
        dist = norm(testPCA(a, :)-trainPCA(1, :));
        for b=1:15*8
            new_dist = norm(testPCA(a, :)-trainPCA(b, :));
            if (new_dist < dist)
                dist = new_dist;
                testIndexes(1, a) = trainIndexes(1, b);
            end
        end
    end

    % Calculation of the nearest-neighbor classification rate:
    Rate = 0;
    for c=1:15*3
        if (testIndexes(1, c) == realTestIndexes(1, c))
            Rate = Rate + 1;
        end
    end
    d = d+1;
    Acc(1, d) = Rate/(15*3);
end

figure;
hold on
plot(1:step:1024, Acc(1, :));
title('Nearest-neighbor classification rate as a function of K');
saveas(gcf, 'RateAsFuncOfK.png');
close(gcf);
hold off