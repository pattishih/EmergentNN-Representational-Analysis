function [returnMatrix,xSortCluster_idxs,ySortCluster_idxs] = classification_clustering (inputData, dataType, numClusters)
% [classifier_sorted, newXIndexes, newYIndexes] = EmProj.classification_clustering(avgLetterTransClassSuccess{layer},'-conmat',7);
DATADIR = '/Volumes/My_HD/Dropbox/matlab/emergentproj/';
load([DATADIR,'type_labels.mat']);

switch lower(dataType)
    case {'-roc' '-performance' '-sensspec' '-sensitivityspecificity' 'roc' 'performance' 'sensspec' 'sensitivityspecificity'}
        %% correlate pattern of sensitivity and specificity
        %layer = 2;
        %numClusters = 8;
        % pc0 = corrcoef(pr_class0_test{layer});
        % pc0_clust = clusterdata(pc0,numClusters);
        % [pc0_clust_sorted,pc0_sortIdx] = sort(pc0_clust);
        % pc0_sorted = corrcoef(pr_class0_test{layer}(:,pc0_sortIdx));
        % figure; imagesc(pc0_sorted,[-1,1]); colorbar
        % set(gca,'xtick',[],'ytick',[])
        % type_labels_sorted = type_labels(pc0_sortIdx,:);

        pcorr0 = corrcoef(inputData);
        pcorr0_clust = clusterdata(pcorr0,numClusters);
        [~,pc0_sortIdx] = sort(pcorr0_clust);
        pcorr0_sorted = corrcoef(pr_class0_test{layer}(:,pc0_sortIdx));
        figure; imagesc(pcorr0_sorted,[-1,1]); colorbar
        set(gca,'xtick',[],'ytick',[])

        %% check for letter matches
        %type_labels:
        %   letter1, ->letter2, 
        %   node1, ->node2, 
        %   letter_transition_type,
        %   node_transition_type
        %1	B
        %2	E
        %3	P
        %4	S
        %5	T
        %6	V
        % %7	X
        % clust_labels = pc0_clust;
        % nLetterMatches = zeros(1,7);
        % for letter = 1:7
        %     
        %     labels_match = (type_labels(:,1)==letter) + (clust_labels==letter);
        %     nLetterMatches(letter) = sum((labels_match==2));
        % end

    case {'-classifier' '-class' 'classifier' 'class'}
        %% correlate classifier
        %layer = 1;
        % %numClusters = 11;
        % cc = corrcoef(classifier{layer});
        % clust_cc = clusterdata(cc,numClusters);
        % [clust_cc_sorted,cc_sortIdx] = sort(clust_cc);
        % classifier_sorted = corrcoef(classifier{layer}(:,cc_sortIdx));
        % figure; imagesc(classifier_sorted,[-1,1]); colorbar
        % set(gca,'xtick',[],'ytick',[])
        % type_labels_sorted = type_labels(cc_sortIdx,:);

        ccorr = corrcoef(inputData);
        ccorr_clust = clusterdata(ccorr,numClusters);
        [~,ccorr_sortIdx] = sort(ccorr_clust);
        classifier_reordered = corrcoef(inputData(:,ccorr_sortIdx));
        figure; imagesc(classifier_reordered,[-1,1]); colorbar
        set(gca,'xtick',[],'ytick',[])

    case {'-conmat' '-confusion' '-confusionmatrix' '-confusionmat' 'confusion' 'confusionmatrix' 'confusionmat' 'conmat'}
    %% correlate confusion matrix classification success for each classifier %%
        %--SET THESE-- %%trial_type: 1, node_type: 2, node1: 3, node2: 4
%         algorithm = 1; %1: mine; 2:Princeton MVPA
%         classificationLowerLimit = 0.0;
%         numClusters = 7;
%
%         classType = 1;
%         processLayers = 2;
%         %-------------
        
%{
        letterTransLabelsTop = {'B->T','B->P','S->S','T->S','T->X','S->X','T->T','P->T','X->T','X->V','T->V','P->V','X->X','P->X','X->S','P->S','V->P','V->V','V->E','S->E'};
        letterTransLabels = {'B->T(N0)','B->P(N0)','S->S(N1)','T->S(N1)','T->X(N1)','S->X(N1)','T->T(N2)','P->T(N2)','X->T(N2)','X->V(N2)','T->V(N2)','P->V(N2)','X->X(N3)','P->X(N3)','X->S(N3)','P->S(N3)','V->P(N4)','V->V(N4)','V->E(N5)','S->E(N5)'};
        nodeTransLabels = {'1:T (N0)','2:P (N0)','3:S (N1)','4:X (N1)','5:T (N2)','6:V (N2)','7:X (N3)','8:S (N3)','9:P (N4)','10:V (N4)','11:E (N5)'};
        nodeLabels = {'N0','N1','N2','N3','N4','N5'};
%}

%         ccorr = corrcoef(avgLetterTransClassSuccess{layer}');
%         ccorr_clust = clusterdata(ccorr,numClusters);
%         [clust_ccorr_sorted,ccorr_sortIdx] = sort(ccorr_clust);
%         classifier_sorted = avgLetterTransClassSuccess{layer}(:,ccorr_sortIdx);%sort columns
%
%         cc2 = corrcoef(classifier_sorted');
%         clust_cc2 = clusterdata(cc2,numClusters);
%         [clust_cc_sorted2,cc_sortIdx2] = sort(clust_cc2);
%         classifier_sorted = classifier_sorted(cc_sortIdx2,:);
%
%         EmProj.confusion_matrix(classifier_sorted,classificationLowerLimit,'Letter Transition','Letter Transition',...
%                         letterTransLabelsTop(ccorr_sortIdx), letterTransLabels(cc_sortIdx2));
% 
%     %    grid_rectangular( 0.5, 20+0.5, 20+1, 0.5, 20+0.5, 20+1 );
%     %    type_labels_sorted = type_labels(cc_sortIdx,:);



        ccorr1 = corrcoef(inputData);
        ccorr1_clust = clusterdata(ccorr1,numClusters);
        [m,n] = size(inputData);
        labelCode = GetLabelCode ( n );

        [~,ccorr1_sortIdx] = sortrows( cat(2,ccorr1_clust,labelCode),[1 2] ); %sort of cluster #, and then sort on node#
        classifier_reordered = inputData(:,ccorr1_sortIdx);%sort columns
%        labelCode_reordered = labelCode(ccorr1_sortIdx,:);
        [m,n] = size(classifier_reordered);
        labelCode = GetLabelCode ( m );
        
        ccorr2 = corrcoef(classifier_reordered');
        ccorr2_clust = clusterdata(ccorr2,numClusters);
        [~,ccorr2_sortIdxs] = sortrows( cat(2,ccorr2_clust,labelCode),[1 2] );
        classifier_reordered = classifier_reordered(ccorr2_sortIdxs,:);%sort rows
        
        returnMatrix = classifier_reordered;
        xSortCluster_idxs = ccorr1_sortIdx; 
%        xSort_type_labels = type_labels(xSortCluster_idxs,:);
        ySortCluster_idxs = ccorr2_sortIdxs; 
%        ySort_type_labels = type_labels(ySortCluster_idxs,:);
        
%        [~,yResortCluster_idxs] = sortrows(ySort_type_labels(:,5),[);
%        EmProj.confusion_matrix(classifier_sorted,classificationLowerLimit,'Letter Transition','Letter Transition',...
%                        newXLabels, newYLabels);
end
end


function labelCode = GetLabelCode ( inputSize )
DATADIR = '/Volumes/My_HD/Dropbox/matlab/emergentproj/';
load([DATADIR,'type_labels.mat']);

        if inputSize == 20
            labelCode = type_labels(:,3);
        elseif inputSize == 11
            labelCode = type_labels2(:,3);
        elseif inputSize == 6
            labelCode = [1:6]';
        end
end