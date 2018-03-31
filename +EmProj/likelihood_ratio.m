function [ratio, sensitivity, specificity] = likelihood_ratio(pr_class1, pr_class0)
% likelihood_ratio  Returns the positive likelihood ratio computed from 
% p(c=1|x) tested on c = {1,0}.
%
% Takes in a vector of p's for class1 and class0 (null class)
% applies a p = .5 cutoff
% counts up the number of 
%       hits vs misses (type II)
%       correct rejects vs false positives (type I)

% can take matricies, but separate tests go in rows and different
% classes go in columns (nTests, nClasses)

%hits
truePos = sum((pr_class1 > .5), 1); %sum across the rows

%misses
falseNeg = sum((pr_class1 <= .5), 1);

%false alarms
falsePos = sum((pr_class0 >= .5), 1);

%correct rejects
trueNeg = sum((pr_class0 < .5), 1);

%sensitivity (true positive rate)
sensitivity = truePos ./ (truePos + falseNeg);

%specificity (true negative rate)
specificity = trueNeg ./ (trueNeg + falsePos);

%positive likelihood ratio:
ratio = sensitivity ./ (1 - specificity);