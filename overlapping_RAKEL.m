function [classifiers,k_labelsets] = overlapping_RAKEL(m, Kl, L, D)
%This is the ensemble production phase of the RAKEL algorithm, with
% overlapping labelsets.
%   It creates an enseble for multi-label classification: each classifier is
%   a single-label LP classifier, constructed considering a random subset of
%   labels, called labelset. In this version of the method, labelsets can
%   be overlapping, this means they can contain common data.

%   The input parameters are defined as follows:
%       m = number of iteration, or in other words, number of classifiers
%       K = dimension of the labelsets
%       L = set of labels
%       D = training set
%   The output parameters represents:
%       classifiers = ensemble of LP classifiers
%       k_labelsets = the randomly selected k-labelsets

% these are all the possible k-labelsets on L
subsets = nchoosek(L.Properties.VariableNames, Kl);
% preallocate output params
k_labelsets = cell(min([m, height(subsets)]), Kl);
classifiers = cell(1, m);
for i=1:height(k_labelsets)
    % randomly select a k-labelset from all the possibilities, and add it
    % to the enseble containing all the selected k-labelsets
    rand = randi([1 height(subsets)],1,1);
    currentLabelset = subsets(rand ,:);
    k_labelsets(i,:) = currentLabelset;
    % remove the selected k-labelset from all the possibilities, to avoid
    % re-selecting it
    subsets(rand, :) = [];
    % prepare Y vector, used in fitcecoc function to train SVM single-label
    % classifiers
    PYi = L(:, currentLabelset);
    Y(1, 1:height(PYi)) = "";
    parfor drug = 1:height(PYi)
        for cls = 1:width(PYi)
            if (PYi{drug, cls} == 1)
                Y(drug) = Y(drug) + PYi.Properties.VariableNames{cls};
            end
        end
    end
    % train a LP classifier hi:X-->P(Yi)on D, where Yi=k-labelset at
    % iteration i.
    % Then, add it to the ensemble of all trained classifiers.
    currentClassifier = fitcecoc(D, Y);
    classifiers{i} = currentClassifier;
end
end