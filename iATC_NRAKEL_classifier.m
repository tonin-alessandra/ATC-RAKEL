clear;
load NRAKEL;
load ATC_42_3883;
% m is the number of iterations, k is the dimension of the labelsets
m=4;
k=4;
% FEAT contains the features associated with each drug
data = array2table(FEAT);
% atcClass contains the labels assigned to each pattern (3883 drugs, each
% one can belong to more than one class).
labels_table = array2table(transpose(atcClass));
% rename colums to better identify the labels
columns = 1:width(labels_table);
newNames = append('c',string(columns));
labels_table = renamevars(labels_table,columns,newNames);
% divide data into training and test set usign 10-fold cross-validation
[trainIndexes, testIndexes] = k_fold(height(data), 10);
% apply the RAKEL algorithm to train single-label SVM classifiers, for each
% fold
parfor tr_fld =1:width(trainIndexes)
    [classifiers_ens{tr_fld}, labelset_set{tr_fld}] = ...
        overlapping_RAKEL(m,k,labels_table(trainIndexes(:,tr_fld), :), ...
        data(trainIndexes(:,tr_fld), :));
end
% apply the RAKEL algorithm to classify new patterns of the test set, for
% each fold
parfor te_fld=1:width(testIndexes)
    [class_vector{te_fld}, negloss{te_fld}, svm_scores{te_fld}] = ...
        test_RAKEL(classifiers_ens{te_fld}, labelset_set{te_fld}, ...
        data(testIndexes(:,te_fld), :), labels_table(testIndexes(:,te_fld), :) );
end

% put together all the results of the folds, in a unique table of predicted
% label, for each drug in the dataset
for f = 1:10
    predicted(testIndexes(:, f), :) = class_vector{f};
end

%performance indicators    
[Absolute_false,Coverage,Absolute_true,Aiming,Accuracy] = ...
    multi_labe_metrics(atcClass,transpose(predicted)); 
