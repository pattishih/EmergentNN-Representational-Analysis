%function NNClass_LogRegr_princeton
% Run classification using logistic regr from Princeton's MVPA package
% i.e., logistic regression optimized by MAP


cd('~/Dropbox/matlab/emergentproj/data');

%--SET THESE:
classType = 3; %1 = letterTrans; 3 = node1
%--
idList = [11];%
grpList = [2];%[1:2]
%--
nFeats = 30;
lamda = nFeats;
%--
verbose = 0;

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
            error('EmProj:LogRegr:classType','Not a valid classType.');
    end

    
%% Setup training data    
    
    dataLabels = importdata([groups{grp},sprintf('%02d',netId),'_trial_labels.txt']);
    dataLabels(:,6) = dataLabels(:,3) + 1; %shift it up by 1 to avoid having 0s
    dataLabels(:,7) = dataLabels(:,4) + 1; %shift it up by 1 to avoid having 0s
    network = importdata([groups{grp},sprintf('%02d',netId),'_trial_layers.txt']);

    [mTrials, ~, ~] = size(network);
    act{1} = network(:,1:30);%hidden_act --> 6x5 = 30 features
    act{2} = network(:,31:60);%context_act --> 6x5 = 30 features

    for layer = 1:2
        meanAct{layer} = mean(act{layer},1);
        for n = 1:nFeats
            centered_act{layer}(:,n) = act{layer}(:,n) - meanAct{layer}(n);
        end
    end


    %%
    clearvars trainingIndexes testingIndexes

    %% Declarations and assignments
    %classifier_w = cell(1,2);
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

    for layer = 1:2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %    classifier_w{layer} = zeros(nFeats,nClasses);
        classifier_b{layer} = zeros(1,nClasses);
        classifier{layer} = zeros(nFeats,nClasses);

        rng('shuffle')

        %Find the fewest number of trials
        nFewestTrials = min(classSizes);
        nHalfTrials = floor(nFewestTrials/2);

        for jClass = 1:nClasses
            %Declare & assign vars
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


        %%
        for jClass = 1:nClasses
            %%
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

            n1 = nHalfTrials;            
            n0 = round(nHalfTrials*2);
            n0_full = length(nullClassTrialIndexes); %779
            
            % training data for class 1:
            x1 = [train_act{layer,jClass}]';%(nFeats, nTrials)

            y1 = ones(1,n1); %binary outcomes -- class1
            
            % training data for class 0:
            x0_full = cat(1,train_act{layer,nullClassIdxSelection})';%large N0 (nFeats, nTrials)
            y0_full = zeros(1,n0_full); %binary outcomes -- class0
            
            
            
            
            
            nRandSamples = 24;
            
            nullRandTrialIndexes{layer,jClass} = zeros(nRandSamples,n0);
            
            %% random sampling from x0_full
            out = cell(1,nRandSamples);
            weights = zeros(nFeats,nRandSamples);
            for iSample = 1:nRandSamples
                if nRandSamples == 1
                    x0 = x0_full;
                    y0 = y0_full;
                else
                   [~, randomizeIdx] = sort(rand(1,n0_full));
                   nullRandTrialIndexes{layer,jClass}(iSample,:) = randomizeIdx(1:n0);
                   x0 = x0_full(:,nullRandTrialIndexes{layer,jClass}(iSample,:));
                   y0 = zeros(1,n0);
                end

               x = cat(2,x1,x0);
               y = cat(2,y1,y0);
             
               out{iSample} = EmProj.PrincetonLogRegrFun(y,x,lamda);
               weights(:,iSample) = out{iSample}.weights;
            end%randSampling
            
            
    %==========================================================================
            par_w = mean(weights,2);
            classifier{layer}(:,jClass) = par_w;

            %% calculate the probabilities p(c=1|x) for the training data :
            % aka test the trained model on the original data (should be good!)
            pr_class1{layer}(:,jClass) = EmProj.LogRegrFun( zeros(1,n1), par_w, x1 );
            pr_class0{layer}(:,jClass) = EmProj.LogRegrFun( zeros(1,n0_full), par_w, x0_full );
            
            if verbose
                disp ' ';
                disp( ['p(c=1|x) for class 1 training data(',num2str(layer),',',num2str(jClass),') = '] );
                fprintf(' %0.3f', pr_class1{layer}(:,jClass));
                disp ' ';
                disp( ['p(c=1|x) for class 0 training data(',num2str(layer),',',num2str(jClass),') = '] );
                    
                fprintf(' %0.3f', pr_class0{layer}(:,jClass));
                disp ' ';
                disp( ['Trained weight coef = ',sprintf(' %0.3f', par_w )]);
                disp '*********************************************';
            end
        end
    end

    %% TEST CLASSIFIERS
    for layer = 1:2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
            pr_class1_test{layer}(:,jClass) = EmProj.LogRegrFun( zeros(1,n1), par_w, x1_test );
            pr_class0_test{layer}(:,jClass) = EmProj.LogRegrFun( zeros(1,n0_full_test), par_w, x0_full_test );
            
            if verbose            
                disp ' ';
                disp([ 'p(c=1|x) for class 1 TEST data(',num2str(layer),',',num2str(jClass),') = '] );
                fprintf(' %0.3f', pr_class1_test{layer}(:,jClass));
                disp ' ';
                disp( ['p(c=1|x) for class 0 TEST data(',num2str(layer),',',num2str(jClass),') = '] );
                fprintf(' %0.3f', pr_class0_test{layer}(:,jClass));
            end
    %%
        end
    end
    disp(['Saving results for ',groups{grp},sprintf('%02d',netId),'...'])
    save([groups{grp},sprintf('%02d',netId),'_',filePrefix,'_Presults.mat'],'train_act','test_act','train_indexes','test_indexes','classSizes','classifier','classifier_b','train_labels','test_labels','pr_class1','pr_class0','pr_class1_test','pr_class0_test')
    disp ' ';
end
end

%==========================================================================

% %%
% figure; imagesc(reshape(classifier_w(:,3),6,5),[-30,17]); colorbar
% cc = corr(classifier_w);
% colormap('jet')
% figure; imagesc((cc>.5).*cc,[0,1]); colorbar


        %     x = linspace(1,tr-1,tr-1);
        %     figure; hold on
        %     plot(x,avgCoef(1,:),'-','Color',[0 0 .8],'LineWidth',2);
        %     plot(x,avgCoef(2,:),'-','Color',[0 .6 0],'LineWidth',2);
        %     hold off
