function likelihood_out = LogRegrFun (par_b, par_w, trainingData, classId)
    switch nargin
        case 3
            % p(c=1|x)
            % likelihood for each data point of being in class1 (given parameters b & w):
            likelihood_out = 1 ./ (1 + exp(  -( par_b + (par_w' * trainingData) )  ));
        case 4 
            % distance from the decision boundary (ie., error)
            likelihood_out = classId - ...
                        (1 / (1 + exp(  -( par_b + (par_w' * trainingData) )  )));
        otherwise
            error('Not enough input arguments.')
    end
end