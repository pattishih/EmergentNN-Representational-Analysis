 
cd('~/Dropbox/matlab/emergentproj/data');

%--SET THESE:
classType = 1; %1 = letterTrans; 3 = node1
%--
idList = [1:10];%simulation# - [1:10]
grpList = [1];%see groups var - [1 2 3]
%--
nFeats = 30;
%--
verbose = 0;%0, 1, or 2

%==========================================================================
groups = {'network' 'control' 'nullcontrol'};
for grp = grpList    
    
    centered_act = cell(1,length(idList));
    jthClassIndexes = cell(1,length(idList));
    for netId = idList
     clearvars -except classType ...
        grpList idList...
        nFeats lamda verbose groups...
        netId grp centered_act jthClassIndexes...
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


    %% Setup data    
        dataLabels = importdata([groups{grp},sprintf('%02d',netId),'_trial_labels.txt']);
        dataLabels(:,6) = dataLabels(:,3) + 1; %shift it up by 1 to avoid having 0s
        dataLabels(:,7) = dataLabels(:,4) + 1; %shift it up by 1 to avoid having 0s
        network = importdata([groups{grp},sprintf('%02d',netId),'_trial_layers.txt']);

        [mTrials, ~, ~] = size(network);
        act{1} = network(:,1:30);%hidden_act --> 6x5 = 30 features
        act{2} = network(:,31:60);%context_act --> 6x5 = 30 features

        % Demean
        meanAct = cell(1,2);
        centered_act{netId} = cell(1,2);
        for layer = 1:2

            centered_act{netId}{layer} = zeros(size(act{layer},1),nFeats);            
            meanAct{layer} = mean(act{layer},1);
            
            for n = 1:nFeats
                centered_act{netId}{layer}(:,n) = act{layer}(:,n) - meanAct{layer}(n);
            end
        end
        
        jthClassIndexes{netId} = cell(1,nClasses);
        for jClass = 1:nClasses
            jthClassIndexes{netId}{jClass} = find(dataLabels(:,classType)==jClass);
        end
    end
    
    
    avgAct_indiv = cell(1,2);
    avgAct_grp = cell(1,2);
    avgActInColumns_grp = cell(1,2);
    treeAvgAct_grp = cell(1,2);
    for layer = 1:2
        for netId = idList
        
            avgAct_indiv{layer} = zeros(30,nClasses,length(idList));
            for jClass = 1:nClasses                
                avgAct_indiv{layer}(:,jClass,netId) = ...
                    mean(centered_act{netId}{layer}(jthClassIndexes{netId}{jClass},:));
            end
        end
        
        avgAct_grp{layer} = mean(avgAct_indiv{layer},3);
        avgActInColumns_grp{layer} = avgAct_grp{layer}';
        treeAvgAct_grp{layer} = linkage(avgActInColumns_grp{layer},'single','euclidean');
        
        letterTransLabels = {'B->T(N0)','B->P(N0)','S->S(N1)','T->S(N1)','T->X(N1)','S->X(N1)','T->T(N2)','P->T(N2)','X->T(N2)','X->V(N2)','T->V(N2)','P->V(N2)','X->X(N3)','P->X(N3)','X->S(N3)','P->S(N3)','V->P(N4)','V->V(N4)','V->E(N5)','S->E(N5)'};
        
        figure; dendrogram(treeAvgAct_grp{layer},'orientation','left','colorthreshold','default','labels',letterTransLabels);
        thickplotline
    end
end
