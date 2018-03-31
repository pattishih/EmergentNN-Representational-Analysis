%function NNClass_LogRegr_ll
% Logistic Regression Using MLE

%%
nCpus = 4;
if ~matlabpool('size')
     matlabpool(nCpus)
end    
 
cd('~/Dropbox/matlab/emergentproj/data');

%--SET THESE:
classType = 3; %1 = letterTrans; 3 = node1
%--
idList = [1:10];%simulation# - [1:10]
grpList = [1:3];%see groups var - [1 2 3]
%--
nFeats = 30;
%--
verbose = 0;%0, 1, or 2

%==========================================================================
groups = {'network' 'control' 'nullcontrol'};
for grp = grpList
    for netId = idList
     clearvars -except classType ...
        grpList idList...
        nFeats lamda verbose groups...
        netId grp ...
        nCpus nNets   

        switch classType
            case 1 %letter transitions
                nClasses = 20;
                filePrefix = 'letterTrans';
            case 2 %node transitions
                nClasses = 11;
                filePrefix = 'nodeTrans';
            case {3,6}
                classType = 6;
                nClasses = 6;
                filePrefix = 'node1';
            case {4,7}
                classType = 7;
                nClasses = 6;
                filePrefix = 'node2';
            otherwise
                error('Not a valid classType.');
        end


    %% Setup training data    
        dataLabels = importdata([groups{grp},sprintf('%02d',netId),'_trial_labels.txt']);
        dataLabels(:,6) = dataLabels(:,3) + 1; %shift it up by 1 to avoid having 0s
        dataLabels(:,7) = dataLabels(:,4) + 1; %shift it up by 1 to avoid having 0s
        network = importdata([groups{grp},sprintf('%02d',netId),'_trial_layers.txt']);

        [mTrials, ~, ~] = size(network);
        act{1} = network(:,1:30);%hidden_act --> 6x5 = 30 features
        act{2} = network(:,31:60);%context_act --> 6x5 = 30 features

        % Demean
        meanAct = cell(1,2);
        centered_act = cell(1,2);
        for layer = 1:2
            meanAct{layer} = mean(act{layer},1);
            for n = 1:nFeats
                centered_act{layer}(:,n) = act{layer}(:,n) - meanAct{layer}(n);
            end
        end

        %% Declarations and assignments
        classifier_b = cell(1,2);
        classifier = cell(1,2);
        train_indexes = cell(2,nClasses);
        test_indexes = cell(2,nClasses);
        train_labels = cell(2,nClasses);
        test_labels = cell(2,nClasses);
        train_act = cell(2,nClasses);
        test_act = cell(2,nClasses);

        classSizes = zeros(1,nClasses);
        jthClassIndexes = cell(1,nClasses);


        %% Split the data in half
        for jClass = 1:nClasses
            jthClassIndexes{jClass} = find(dataLabels(:,classType)==jClass);
            classSizes(jClass) = length(jthClassIndexes{jClass});
        end
        
        rng('shuffle')
        for layer = 1:2
            classifier_b{layer} = zeros(1,nClasses);
            classifier{layer} = zeros(nFeats,nClasses);

            % Find the fewest number of trials of all classes
            nFewestTrials = min(classSizes);
            nHalfTrials = floor(nFewestTrials/2);

            for jClass = 1:nClasses
                % Declare & assign vars
                train_indexes{layer,jClass} = zeros(nHalfTrials,1);
                test_indexes{layer,jClass} = zeros(nHalfTrials,1);

                permutedIndex_holder = randperm(classSizes(jClass));
                randSelect_jthClassIdx_train = permutedIndex_holder(1:nHalfTrials)';%transpose to keep things consistent --> trials in rows
                randSelect_jthClassIdx_test = permutedIndex_holder(end-nHalfTrials+1:end)';%transpose to keep things consistent --> trials in rows

                train_indexes{layer,jClass} = jthClassIndexes{jClass}(randSelect_jthClassIdx_train);
                test_indexes{layer,jClass} = jthClassIndexes{jClass}(randSelect_jthClassIdx_test);

                train_labels{layer,jClass} = dataLabels(train_indexes{layer,jClass},:); %not used
                test_labels{layer,jClass} = dataLabels(test_indexes{layer,jClass},:); %not used
                train_act{layer,jClass} = centered_act{layer}(train_indexes{layer,jClass},:);
                test_act{layer,jClass} = centered_act{layer}(test_indexes{layer,jClass},:);   
            end


            %% Train classifiers for each class
            for jClass = 1:nClasses
                switch jClass
                    case 1
                        nullClassIdxSelection = 2:nClasses;
                    case nClasses
                        nullClassIdxSelection = 1:nClasses-1;
                    otherwise
                        nullClassIdxSelection = [1:jClass-1,jClass+1:nClasses];
                end

                classTrialIndexes = train_indexes{layer,jClass};  
                nullClassTrialIndexes = cat(1,train_indexes{layer,nullClassIdxSelection});

                % training data for class 1:
                x1 = [train_act{layer,jClass}]';%trials now in cols; feats in rows

                % training data for class 0:
                x0_full = cat(1,train_act{layer,nullClassIdxSelection})';%large N0 trials in cols; feats in rows

                % get class sample sizes
                n1 = nHalfTrials;
                n0 = round(nHalfTrials*2);
                n0_full = length(nullClassTrialIndexes); %779
                
                % Set number of random samples of class0 trials of size n0
                nRandSamples = 12;

                % variable declarations
                nullRandTrialIndexes = zeros(nRandSamples,n0);
                par_w_randSampling = zeros(30,nRandSamples);
                par_b_randSampling = zeros(1,nRandSamples);
                x0 = cell(1,nRandSamples);
                
                %==========================================================
                %% Slice up the data for parallel computing
                for subslice = 1:nRandSamples
                    [~, randomizeIdx] = sort(rand(1,n0_full));
                    nullRandTrialIndexes = randomizeIdx(1:n0);
                    x0{subslice} =  x0_full(:,nullRandTrialIndexes);
                end    

                x1_sliced = cell(1,nCpus);
                for prepool = 1:nCpus
                    x1_sliced{prepool} = x1;
                    sampleSets = prepool:nCpus:nRandSamples;
                    for ss = 1:length(sampleSets)
                        x0_sliced{prepool}{ss} = x0{sampleSets(ss)};
                    end
                end
                
                %% Run each data-slice on a separate CPU
                parfor pool = 1:nCpus
                nLoops = numel(x0_sliced{pool});

                for subslice = 1:nLoops
                    % Initial guess about the parameters:
                    par_b = 0; % bias
                    par_w = zeros(nFeats,1); %weight vector

                    % Max number of updating (training) iterations:
                    tr = 0; trMax = 2000;

                    % Learning rate:
                    eta = 5/(n1+n0);
                    thresh = 0.1;

                    threshHolder = [];
                    maxNum = 20;
                    maxUpdates = maxNum;
                    opt_param = zeros(1,maxUpdates);
                    grandTr = 0;

                    % Set initial likelihood to a "large" num to enter loop
                    L = 1;
                    noreset = 0;
                    
                    % set gradients initally to ensure that we enter the update loop:
%                    gradient_b = 1; gradient_w = zeros(size(par_w));                    
%                    while sum(abs(gradient_w)) + abs(gradient_b) > 0.1; %used batched update gradient ascent
                    
                    while L > thresh % continues while change in weights is still large...
                        tr = tr + 1;  % increment the number of updates carried out    

                        % reset gradients to zero
%                        gradient_b = 0; gradient_w = 0*gradient_w;
                        L = 0;
                        Sum_L_err = 0;

                        %% Cycle through class 1 trials (cols):
                        % Goal here is to minimize the amount of false
                        % negatives (type II error; beta)
                        for u = 1:n1
                            % class 1:
                            % (returns decision error given current parameters) 
                            L_err = EmProj.LogRegrFun( par_b, par_w, x1_sliced{pool}(:,u), 1 );

                            Sum_L_err = Sum_L_err + L_err;

%                             gradient_b = gradient_b + L_err; %summed deviations from the decision boundary
%                             gradient_w = gradient_w + L_err * x1(:,u);

                            % Online updating of parameters:
                            par_b = par_b + (eta * L_err); % update bias scalar by learning rate 'eta'
                            par_w = par_w + (eta * L_err * x1_sliced{pool}(:,u)); % update weight vector by learning rate 'eta'
                        end


                        %% Cycle through class 0 trials (cols):
                        % Goal here is to minimize the amount of false positives (type I error; alpha)
                        for c0 = 1:1
                        for u = 1:n0
                            % class 0:
                            % (returns decision error given current parameters) 
                            L_err = EmProj.LogRegrFun( par_b, par_w, x0_sliced{pool}{subslice}(:,u), 0 );

                            Sum_L_err = Sum_L_err + L_err;
                             
%                             gradient_b = gradient_b + L_err;
%                             gradient_w = gradient_w + L_err * x0_sliced{pool}{subslice}(:,u);

                            % Online updating of parameters:
                            par_b = par_b + (eta * L_err); % update bias scalar by learning rate 'eta'
                            par_w = par_w + (eta * L_err * x0_sliced{pool}{subslice}(:,u)); % update weight vector by learning rate 'eta'
                        end
                        end
                        
                        %% Batch updating of the parameters:
%                         par_b = par_b + (eta * gradient_b); % update bias scalar by learning rate 'eta'
%                         par_w = par_w + (eta * gradient_w); % update weight vector by learning rate 'eta'

                        L = abs(Sum_L_err);
                        opt_param(tr) = L;

                        grandTr=grandTr+1;
                        if tr == 1 && verbose
                            disp(['optimization term(',num2str(pool),',',num2str(layer),',',num2str(jClass),'): ',mat2str(opt_param(1))]);
                        elseif tr > maxUpdates
                            %Look for any minima in optimization
                            diff1 = diff((diff(opt_param,2)>0.1));
                            diff2 = (diff1>0);
                            numMinima = sum((diff2==1));

                            if ~noreset %if noreset has not been set
                                par_b = 0; % bias
                                par_w = zeros(nFeats,1); %weight vector
                                tr = 0;
                                opt_param = [];

                                if verbose == 2
                                    disp(['n minima = ',num2str(numMinima)']);
                                    disp '...Resetting...';
%                                      disp(['optimization term(',num2str(pool),',',num2str(layer),',',num2str(jClass),'): ',mat2str(opt_param(tr-8))]);
%                                      disp(['optimization term(',num2str(pool),',',num2str(layer),',',num2str(jClass),'): ',mat2str(opt_param(tr-6))]);
%                                      disp(['optimization term(',num2str(pool),',',num2str(layer),',',num2str(jClass),'): ',mat2str(opt_param(tr-4))]);
%                                      disp(['optimization term(',num2str(pool),',',num2str(layer),',',num2str(jClass),'): ',mat2str(opt_param(tr-2))]);
                                    disp(['optimization term(',num2str(pool),',',num2str(layer),',',num2str(jClass),'): ',mat2str(opt_param(tr))]);
                                end                            
                            else
                                if tr > maxUpdates*2 && verbose == 2
                                    disp(['optimization term(',num2str(pool),',',num2str(layer),',',num2str(jClass),'): ',mat2str(opt_param(tr))]);
                                end
                            end
                            if numMinima > 0

                                if numMinima/maxUpdates >= 0.5
                                    eta = eta/1.2;      
                                    threshHolder = min([opt_param threshHolder]);       
                                    maxUpdates = maxNum;
                                    if verbose == 2
                                        disp(['> current thresh = ',num2str(thresh),'; large lrate chg = ',num2str(eta)]);
                                    end
                                elseif numMinima/maxUpdates >= 0.3
                                    eta = eta/1.1;
%                                     threshHolder = min([opt_param threshHolder]);  
                                    if grandTr > 1000
                                        grandTr = 0;
                                        thresh = min([opt_param threshHolder]);
                                        threshHolder = thresh;
                                    else
                                        threshHolder = min([opt_param threshHolder]);
                                    end

                                    maxUpdates = maxNum;
                                    if verbose == 2
                                        disp(['> current thresh = ',num2str(thresh),'; med lrate chg = ',num2str(eta)]);
                                    end  
                                else
                                    eta = eta/1.05;
%                                    threshHolder = min([opt_param threshHolder]);  
                                    if grandTr > 1000
                                        grandTr = 0;
                                        thresh = min([opt_param threshHolder]);
                                        threshHolder = thresh;
                                    else
                                        threshHolder = min([opt_param threshHolder]);
                                    end

                                    maxUpdates = maxNum;
                                    if verbose == 2
                                        disp(['> current thresh = ',num2str(thresh),'; small lrate chg = ',num2str(eta)]);
                                    end
                                end
                            else
                                noreset = 1;
%                                 eta = eta/1.05;
                                maxUpdates = 100;
                            end

                            if grandTr > 1000
                                grandTr = 0;
                                thresh = min([opt_param threshHolder]);
                                threshHolder = thresh;
                            else
                                threshHolder = min([opt_param threshHolder]);
                            end
                        elseif tr == trMax
                            if verbose
                                disp('>>> ',[num2str(trMax),' iterations reached. Break.']);
                            end
                            break;
                        else
                            continue;
                        end
                    end

                    if verbose
                        disp(['optimization term(',num2str(pool),',',num2str(layer),',',num2str(jClass),'): ',mat2str(opt_param(tr))]); 
                    end

                    disp(['>>> Finished optimizing(',num2str(pool),',',num2str(layer),',',num2str(jClass),') in ',num2str(tr),' iterations.']); 
                    disp ' ';
                    par_w_randSampling(:,pool) = par_w;
                    par_b_randSampling(1,pool) = par_b;
                end%randSampling
                end%pool
                %==========================================================
                
                % Average the trained parameter weights across class0 subsamples
                par_w = mean(par_w_randSampling,2);
                par_b = mean(par_b_randSampling,2);

                %% calculate the final probabilities p(c=1|x) for the training data :
                %-- test original data on trained model
                pr_class1{layer}(:,jClass) = EmProj.LogRegrFun( repmat(par_b,1,n1), par_w, x1 );
                pr_class0{layer}(:,jClass) = EmProj.LogRegrFun( repmat(par_b,1,n0_full), par_w, x0_full );

                if verbose
                    disp ' ';
                    disp( ['p(c=1|x) for class 1 training data(',num2str(layer),',',num2str(jClass),') = '] );
                        fprintf(' %0.3f', pr_class1{layer}(:,jClass));
                    disp ' ';
                    disp( ['p(c=1|x) for class 0 training data(',num2str(layer),',',num2str(jClass),') = '] );        
                        fprintf(' %0.3f', pr_class0{layer}(:,jClass));

                    disp ' ';
                    disp( ['Trained weight coef = ',sprintf(' %0.3f', par_w )]);
                    disp( ['Trained bias coef = ',sprintf(' %0.3f', par_b )]);
                    disp '*********************************************';
                end
                classifier{layer}(:,jClass) = par_w;
                classifier_b{layer}(1,jClass) = par_b;
            end
        end

        %% TEST CLASSIFIERS
        for layer = 1:2
            for jClass = 1:nClasses

                switch jClass
                    case 1
                        nullClassIdxSelections = 2:nClasses;
                    case nClasses
                        nullClassIdxSelections = 1:nClasses-1;
                    otherwise
                        nullClassIdxSelections = [1:jClass-1,jClass+1:nClasses];
                end

                par_w = classifier{layer}(:,jClass);

                x1_test = [test_act{layer,jClass}]';

                x0_full_test = cat(1,test_act{layer,nullClassIdxSelections})';%large N0 (nFeats, nTrials)
                [~, n0_full_test] = size(x0_full_test); %should be 779


                %% Calculate the probabilities p(c=1|x) for the test data :
                pr_class1_test{layer}(:,jClass) = EmProj.LogRegrFun( repmat(par_b,1,n1), par_w, x1_test );
                pr_class0_test{layer}(:,jClass) = EmProj.LogRegrFun( repmat(par_b,1,n0_full_test), par_w, x0_full_test );

                if verbose
                    disp ' ';
                    disp([ 'p(c=1|x) for class 1 TEST data(',num2str(layer),',',num2str(jClass),') = '] );
                        fprintf(' %0.3f', pr_class1_test{layer}(:,jClass));
                    disp ' ';
                    disp( ['p(c=1|x) for class 0 TEST data(',num2str(layer),',',num2str(jClass),') = '] );
                        fprintf(' %0.3f', pr_class0_test{layer}(:,jClass));
                end
            end
        end
        
        %% Save the results
        disp(['Saving results for ',groups{grp},sprintf('%02d',netId),'...'])
        save([groups{grp},sprintf('%02d',netId),'_',filePrefix,'_results.mat'],'train_act','test_act','train_indexes','test_indexes','classSizes','classifier','classifier_b','train_labels','test_labels','pr_class1','pr_class0','pr_class1_test','pr_class0_test');
        disp ' ';
    end%netId
end%grp

%matlabpool close

