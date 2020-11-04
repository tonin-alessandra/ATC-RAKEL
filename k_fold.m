% Organize data into training and test set for cross-validation.
%   The output are the indexes of the training and the test set for each
%   fold.

function [trainIdx, testIdx] = k_fold(N,kf)
kfolds = cvpartition(N, 'KFold', kf);
% training and test indexes
trainIdx = zeros(N, kfolds.NumTestSets, 'logical');
testIdx = zeros(N, kfolds.NumTestSets, 'logical');
for fold = 1:kfolds.NumTestSets
    trainIdx(:, fold) = training(kfolds, fold);
    testIdx(:, fold) = test(kfolds, fold);
end
end