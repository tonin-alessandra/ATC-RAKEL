function [result_vector, negloss, svm_scores] = test_RAKEL(H, Y, X, L)
%This is the ensemble combination phase of the RAKEL algorithm.
%   To classify a new pattern x, each model h provides binary decision for
%   each label y in the k-labelset Yi.
%   Then, for each label the average decision is calculated for each label
%   in L and takes a positive decision in this value is grater than a
%   threshold t.
%
%   The input parameters are defined as follows:
%       H = ensemble of the single-label classifiers for current fold
%       Y = set of randomly selected k-labelsets for current fold
%       X = test set
%       L = set of labels
%   The output parameter are:
%       result_vector = the multilabel classification vector
%       negloss = negated average binary losses, returned as nxl matrix
%                 where n is the number of observations and l is the number
%                 of distinct classes in the training data.
%       svm_scores = positive-class scores for each binary learner,
%                    returned as nxB matrix, where n is the number of 
%                    observations and B is the number of binary learners.
%                    
%   For other information about outputs, please refer to 
%   https://it.mathworks.com/help/stats/classificationecoc.predict.html
% 
% Notice that before calling this function, overlapping_RAKEL must be called,
% in order to create the ensemble of classifiers and to select the k-labelsets.

t = 0.5; % threshold

% each classifier must provide a binary decision for the new pattern
predLabel = cell(height(X),size(H,2));
negloss = cell(1,size(H,2));
svm_scores = cell(1,size(H,2));
for svm_mdl = 1:size(H,2)
        [predLabel(:, svm_mdl),negloss{svm_mdl}, svm_scores{svm_mdl}] = predict(H{svm_mdl}, X, 'ObservationsIn', 'rows');
end
for row = 1:size(predLabel,1)
    % create a temporary table for each pattern of this fold as follows:
    %   - rows represents the m classifiers
    %   - columns represents the 14 classes of ATC classification
    %   - each element can be 1 (if the correspondent class belongs to the
    %     current labelset on which the classifier was trained, and the
    %     classifier decided that this drug belongs to this class), 0 (if the
    %     correspondent class belongs to the current labelset on which the
    %     classifier was trained, but the classifier decided that this drug
    %     DOESN'T belong to this class) or NaN (if the correspondent class
    %     DOESN'T belong to the current labelset). This last one is only a
    %     sentinel value to avoid mistakes in the computation of votes and
    %     final decision of the ensemble of classifiers.
    for clsf=1:size(predLabel,2)
        for atccls = 1:width(L)
            if(any(strcmp("c"+string(atccls),Y(clsf,:))))
                temp(clsf, atccls) = 0;
                if ((contains(predLabel(row, clsf), "c"+string(atccls)+"c")) ...
                        || endsWith(predLabel(row, clsf), "c"+string(atccls)))
                    temp(clsf, atccls) = 1;
                end
            else
                temp(clsf, atccls) = NaN;
            end
        end
    end
    % for each pattern, calculate the final decision vector, as the ratio 
    % between the sum of positive decisions (1's) and total number of votes 
    % (elements different from NaN) from all the classifiers.
    % Then, this ratio is compared with the threshold value to decide if
    % the pattern belongs to the class or not.
    for c = 1:width(L)
        result_vector(row, c) = ((sum(temp(:,c)==1))/(sum(temp(:,c)==0) + sum(temp(:,c)==1))) >= t;
    end
end
% to indicate that a pattern doesn't belong to a certain class, switch from
% 0 to -1 to be consistent with the notation in the dataset
result_vector = double(result_vector);
result_vector(result_vector == 0) = -1;
end