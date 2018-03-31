 %%
%cd('/Volumes/My_HD/Dropbox/matlab/emergentproj');
cd('~/Dropbox/matlab/emergentproj/data');


%--SET THESE--
alg = 1; %1: mine; 2:Princeton MVPA

grpList = [1];%loopable [1 2 3]: 1(trained); 2(control); 3(control II)
classList = [3]; %loopable: 1(letterTrans); 3(node1)
%---
processLayers = [2]; %loopable [1 2]
%---
computePerformance = 1;
plotResults = 1;

%- plotting options
pauseOn = 0;
classificationLowerLimit = 0.0;
sortResults = [0]; %loopable [0 1]
popup = 1;
savePlots = 0;
%-------------
%plotClassifiers = 0; %NOT USEFUL bc networks can use different patterns to successfully classify
%idList = [1:10]; %loopable


%==========================================================================
for grp = grpList
    if grp < 3
        idList = 1:10;
    else
        idList = 11;
    end
    
for classType = classList
    
    clearvars -except   alg idList grpList classList processLayers...
                        computePerformance plotResults plotClassifiers...
                        popup pauseOn ...
                        classificationLowerLimit...
                        sortResults savePlots...
                        grp grpList classType classList *_save
                          
    groups = {'network' 'control' 'nullcontrol'};
    nFeats = 30;
    nNets = length(idList);
    
    switch alg
        case 1
            resultsPrefix = 'results';
            pngPrefix = '.png';

        case 2
            resultsPrefix = 'Presults';
            pngPrefix = '_princeton.png';
    end

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

    nLetterTrans = 20;
    nNodeTrans = 11;
    nNodes = 6;

    classcorr = cell(2,nNets);
    letterTransClassSuccess = cell(1,2);
    sseCol = 5;
    classifierMats = cell(1,2);
    test_labels_mat = cell(nNets,2);
    for layer = processLayers
        letterTransClassSuccess{layer} = zeros(nLetterTrans,nClasses,nNets);
        
        for netId = idList
            %test_labels: 2x6 (layer x class)
            networks(netId) = load([groups{grp},sprintf('%02d',netId),'_',filePrefix,'_',resultsPrefix,'.mat']);

            [~,nDataColumns] = size(networks(netId).test_labels{1});

%            classcorr{netId,layer} = corr( networks(netId).classifier{layer} );
            % Get network classifiers
            [mFeatures, nClassifiers] = size(networks(netId).classifier{layer});
            classifierMats{layer}(netId,:,:) = zeros(mFeatures,nClassifiers);
            classifierMats{layer}(netId,:,:) = networks(netId).classifier{layer};

            pr_class1_test = networks(netId).pr_class1_test{layer}; %183x6
            pr_class0_test = networks(netId).pr_class0_test{layer}; %915x6

            n1 = length(pr_class1_test);
            n0 = length(pr_class0_test);

            layerLower = 1 + n1*(layer-1);
            layerUpper = n1 + n1*(layer-1);


            for jClassifier = 1:nClasses

                % Get performance of classifiers
                if computePerformance
                    performance{layer}(1,jClassifier,netId) = (sum(pr_class1_test(:,jClassifier) > 0.5) + sum(pr_class0_test(:,jClassifier) < 0.5)) / (length(pr_class1_test(:,jClassifier)) + length(pr_class0_test(:,jClassifier)));
                    [performance{layer}(4,jClassifier,netId), performance{layer}(2,jClassifier,netId), performance{layer}(3,jClassifier,netId)] =...
                        EmProj.likelihood_ratio(pr_class1_test(:,jClassifier), pr_class0_test(:,jClassifier));
                end            
                
                
                switch jClassifier
                    case 1
                        nullClasses = 2:nClasses;
                    case nClasses
                        nullClasses = 1:nClasses-1;
                    otherwise
                        nullClasses = [1:jClassifier-1,jClassifier+1:nClasses];
                end
                test_labels_mat{netId,layer} = zeros(n1,nDataColumns,nClasses);%set up empty matrix
                test_labels_mat_tmp = reshape( cell2mat(networks(netId).test_labels), n1*2,nDataColumns,nClasses);
                test_labels_mat{netId,layer} = test_labels_mat_tmp(layerLower:layerUpper,:,:); clearvars *_tmp;

                % Letter transitions:
                letterTransCol = 1;
                letterTrans1_labels = networks(netId).test_labels{layer,jClassifier}(:,letterTransCol);
                letterTrans0_labels = test_labels_mat{netId,layer}(:,letterTransCol,nullClasses);

                for pLetterTrans = 1:nLetterTrans
                    letterTrans1_Idxs = find( letterTrans1_labels == pLetterTrans );
                    letterTrans0_Idxs = find( letterTrans0_labels == pLetterTrans );

                    pr_class_letterTrans = pr_class1_test(letterTrans1_Idxs,jClassifier);
                    pr_class_letterTrans = cat(1, pr_class_letterTrans, pr_class0_test(letterTrans0_Idxs,jClassifier));%combine all probabilities for this class into one column

                    nLetterTransTrials{layer}(pLetterTrans,netId) = length(pr_class_letterTrans); 
                    letterTransClassSuccess{layer}(pLetterTrans,jClassifier,netId) = sum(pr_class_letterTrans > 0.5) / nLetterTransTrials{layer}(pLetterTrans,netId);
                end%letterTrans

                % Node transitions:
                nodeTransCol = 2;
                nodeTrans1_labels = networks(netId).test_labels{layer,jClassifier}(:,nodeTransCol);
                nodeTrans0_labels = test_labels_mat{netId,layer}(:,nodeTransCol,nullClasses);

                for pNodeTrans = 1:nNodeTrans
                    nodeTrans1_Idxs = find( nodeTrans1_labels == pNodeTrans );
                    nodeTrans0_Idxs = find( nodeTrans0_labels == pNodeTrans );

                    pr_class_nodeTrans = pr_class1_test(nodeTrans1_Idxs,jClassifier);
                    pr_class_nodeTrans = cat(1, pr_class_nodeTrans, pr_class0_test(nodeTrans0_Idxs,jClassifier));%combine all probabilities for this class into one column

                    nNodeTransTrials{layer}(pNodeTrans,netId) = length(pr_class_nodeTrans);
                    nodeTransClassSuccess{layer}(pNodeTrans,jClassifier,netId) = sum(pr_class_nodeTrans > 0.5) / nNodeTransTrials{layer}(pNodeTrans,netId);
                end%nodeTrans

                % Node1:
                nodeCol = 6;
                node1_labels = networks(netId).test_labels{layer,jClassifier}(:,nodeCol);
                node0_labels = test_labels_mat{netId,layer}(:,nodeCol,nullClasses);

                for pNode = 1:nNodes
                    node1_Idxs = find( node1_labels == pNode );
                    node0_Idxs = find( node0_labels == pNode );

                    pr_class_nodes = pr_class1_test(node1_Idxs,jClassifier);
                    pr_class_nodes = cat(1, pr_class_nodes, pr_class0_test(node0_Idxs,jClassifier));%combine all probabilities for this class into one column

                    nNodeTrials{layer}(pNode,netId) = length(pr_class_nodes);
                    nodeClassSuccess{layer}(pNode,jClassifier,netId) = sum(pr_class_nodes > 0.5) / nNodeTrials{layer}(pNode,netId);
                end%nodes
            end%class
        end%netId
        if computePerformance, avgPerformance{layer} = mean(performance{layer}(:,:,idList(1):idList(end)),3); stdevPerformance{layer} = std(performance{layer}(:,:,idList(1):idList(end)),0,3); end
        avgLetterTransClassSuccess{layer} = mean(letterTransClassSuccess{layer}(:,:,idList(1):idList(end)),3);
        avgNodeTransClassSuccess{layer} = mean(nodeTransClassSuccess{layer}(:,:,idList(1):idList(end)),3);
        avgNodeClassSuccess{layer} = mean(nodeClassSuccess{layer}(:,:,idList(1):idList(end)),3);
        
        for k = 1:nClasses
            [~,pLetterTransClassSuccess{layer}] = ttest(letterTransClassSuccess{layer}(:,:,idList(1):idList(end)),[],[],'both',3);
        end
          %% Get average classifiers (inactive)
%         if plotClassifiers
%             avgClassifiers_tmp = reshape( mean(classifierMats{layer},1), mFeatures,  nClassifiers );
% 
%             for cl = 1:nClassifiers %TO DO: merge with nClasses later?
%                 avgClassifier{layer,cl} = reshape( avgClassifiers_tmp(:,cl), 6,5 ); %dimensions of each layer (i.e., 6x5)
%             end
%         end
%%
        clearvars *_tmp
    end%layer


%% Plot classifiers (inactive) --------------------------------------------
% The following works, but plotting the average classifier doesn't actually
% make sense, right? However, the code is kept here in case it is needed in
% the future...
%
%     if plotClassifiers               
%         for layer = processLayers
%             figSubplot_h = figure;
%             figSubplot_pos = get(figSubplot_h,'Position');
%             switch nClassifiers
%                 case 20
%                     nRows = 5;
%                     nCols = 4;
%                     className = 'Letter Transition Classifiers';
%                     set(figSubplot_h,'Position',[figSubplot_pos(1),figSubplot_pos(2),figSubplot_pos(3),figSubplot_pos(4)*1.5])
%                     xclassLabels = {'B->T(N0)','B->P(N0)','S->S(N1)','T->S(N1)','T->X(N1)','S->X(N1)','T->T(N2)','P->T(N2)','X->T(N2)','X->V(N2)','T->V(N2)','P->V(N2)','X->X(N3)','P->X(N3)','X->S(N3)','P->S(N3)','V->P(N4)','V->V(N4)','V->E(N5)','S->E(N5)'};
%                 case 11
%                     nRows = 4;
%                     nCols = 3;
%                     className = 'Node Transition Classifiers';
%                     set(figSubplot_h,'Position',[figSubplot_pos(1),figSubplot_pos(2),figSubplot_pos(3),figSubplot_pos(4)*1.2])
%                     xclassLabels = {'1:T (N0)','2:P (N0)','3:S (N1)','4:X (N1)','5:T (N2)','6:V (N2)','7:X (N3)','8:S (N3)','9:P (N4)','10:V (N4)','11:E (N5)'};
%                 case 6
%                     nRows = 2;
%                     nCols = 3;
%                     xclassLabels = {'N0','N1','N2','N3','N4','N5'};
%                     className = 'Node Classifiers';
%                     %set(figSubplot_h,'Position',[figSubplot_pos(1),figSubplot_pos(2),figSubplot_pos(3),figSubplot_pos(4)*1.5])
%             end                   
%             switch layer
%                 case 1
%                     layerName = 'Hidden Layer';
%                 case 2
%                     layerName = 'Context Layer';
%             end
%             for jClassifier = 1:nClassifiers
%                 %subplot (nrows,ncols,plot_number)
%                 cupper = max(max(cell2mat(avgClassifier)));
%                 clower = min(min(cell2mat(avgClassifier)));
%                 splot_h = subplot(nRows,nCols,jClassifier);
%                 imagesc(rot90(avgClassifier{layer,jClassifier}),[clower cupper]);
%                 xlabel(xclassLabels{jClassifier},'FontWeight','bold','FontSize',14);
%                 set(gca, 'XTickLabel', [],'YTickLabel', [])
%                 axis image
%             end
%             
%             ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
%             text(0.55, 1,['\bf ',className,': ',layerName],'HorizontalAlignment','center','VerticalAlignment', 'top','FontSize',18)
% %            subplot(nRows,nCols,cBarLoc,'Visible','off');
% %            colorbar('location','North')
%             if pauseOn,pause; end
%         end
%     end
%%

    letterTransLabelsTop = {'B->T','B->P','S->S','T->S','T->X','S->X','T->T','P->T','X->T','X->V','T->V','P->V','X->X','P->X','X->S','P->S','V->P','V->V','V->E','S->E'};
    letterTransLabels = {'B->T(N0)','B->P(N0)','S->S(N1)','T->S(N1)','T->X(N1)','S->X(N1)','T->T(N2)','P->T(N2)','X->T(N2)','X->V(N2)','T->V(N2)','P->V(N2)','X->X(N3)','P->X(N3)','X->S(N3)','P->S(N3)','V->P(N4)','V->V(N4)','V->E(N5)','S->E(N5)'};
    nodeTransLabels = {'1:T (N0)','2:P (N0)','3:S (N1)','4:X (N1)','5:T (N2)','6:V (N2)','7:X (N3)','8:S (N3)','9:P (N4)','10:V (N4)','11:E (N5)'};
    nodeLabels = {'N0','N1','N2','N3','N4','N5'};
    letterTransReorder = [1 2 3 6 4 5 8 12 7 11 9 10 14 16 13 15 18 17 19 20];
    letterTransLabelsTop = letterTransLabelsTop(letterTransReorder);
    letterTransLabels = letterTransLabels(letterTransReorder);
    avgLetterTransClassSuccess{layer} = avgLetterTransClassSuccess{layer}(letterTransReorder, :);
    pLetterTransClassSuccess{layer} = pLetterTransClassSuccess{layer}(letterTransReorder, :);
    
    if plotResults
        for sortMat = sortResults

            if sortMat
                sortPrefix = 'sorted';
            else
                sortPrefix = 'nosort';
            end   

            
            switch classType
                case 1 %letterTrans classifiers
                    for layer = processLayers
                        classificationMat = cell(1,3);
                        matXLabels = cell(1,3);
                        matYLabels = cell(1,3);
                        
                        avgLetterTransClassSuccess{layer} = avgLetterTransClassSuccess{layer}(:,letterTransReorder);
                        
                        if sortMat
                            [classificationMat{1}, newXIndexes{1}, newYIndexes{1}] = ...
                                        EmProj.classification_clustering(avgLetterTransClassSuccess{layer},'-conmat',6);
                            matXLabels{1} = letterTransLabelsTop(newXIndexes{1});
                            matYLabels{1} = letterTransLabels(newYIndexes{1});

                            [classificationMat{2}, newXIndexes{2}, newYIndexes{2}] = ...
                                        EmProj.classification_clustering(avgNodeTransClassSuccess{layer},'-conmat',6);
                            matXLabels{2} = letterTransLabelsTop(newXIndexes{2});
                            matYLabels{2} = nodeTransLabels(newYIndexes{2});

                            [classificationMat{3}, newXIndexes{3}, newYIndexes{3}] = ...
                                        EmProj.classification_clustering(avgNodeClassSuccess{layer},'-conmat',6);
                            matXLabels{3} = letterTransLabels(newXIndexes{3});
                            matYLabels{3} = nodeLabels(newYIndexes{3});                


                        else
                            classificationMat{1} = avgLetterTransClassSuccess{layer};
                            matXLabels{1} = letterTransLabelsTop;
                            matYLabels{1} = letterTransLabels;

                            classificationMat{2} = avgNodeTransClassSuccess{layer};
                            matXLabels{2} = letterTransLabelsTop;
                            matYLabels{2} = nodeTransLabels;

                            classificationMat{3} = avgNodeClassSuccess{layer};
                            matXLabels{3} = letterTransLabels;
                            matYLabels{3} = nodeLabels;                

                        end
                        figure_h1 = EmProj.confusion_matrix(classificationMat{1},classificationLowerLimit,'Letter Transition','Letter Transition',...
                                matXLabels{1}, matYLabels{1}, popup);            
                        figure_h2 = EmProj.confusion_matrix(classificationMat{2},classificationLowerLimit,'Letter Transition','Node Transition',...
                                matXLabels{2}, matYLabels{2}, popup);
                        figure_h3 = EmProj.confusion_matrix(classificationMat{3},classificationLowerLimit,'Letter Transition','Node',...
                                matXLabels{3}, matYLabels{3}, popup);    

                        if savePlots
                            export_fig(['../','conmat_',groups{grp},'_L',num2str(layer),'_letterTransByLetterTrans_',sortPrefix,pngPrefix],'-transparent','-q101','-m3',figure_h1);
                            export_fig(['../','conmat_',groups{grp},'_L',num2str(layer),'_letterTransByNodeTrans_',sortPrefix,pngPrefix],'-transparent','-q101','-m3',figure_h2);
                            export_fig(['../','conmat_',groups{grp},'_L',num2str(layer),'_letterTransByNodes_',sortPrefix,pngPrefix],'-transparent','-q101','-m3',figure_h3);                             
                        end
                        if length(processLayers) > 1 && pauseOn; pause; end
                    end

                case 2 %node transition classifiers
                case {3,6} %node 1 classifiers
                    for layer = processLayers
                        classificationMat = cell(1,3);
                        matXLabels = cell(1,3);
                        matYLabels = cell(1,3);
                        if sortMat
                            [classificationMat{1}, newXIndexes{1}, newYIndexes{1}] = ...
                                        EmProj.classification_clustering(avgLetterTransClassSuccess{layer},'-conmat',6);
                            matXLabels{1} = nodeLabels(newXIndexes{1});
                            matYLabels{1} = letterTransLabels(newYIndexes{1});

                            [classificationMat{2}, newXIndexes{2}, newYIndexes{2}] = ...
                                        EmProj.classification_clustering(avgNodeTransClassSuccess{layer},'-conmat',6);
                            matXLabels{2} = nodeLabels(newXIndexes{2});
                            matYLabels{2} = nodeTransLabels(newYIndexes{2});

                            [classificationMat{3}, newXIndexes{3}, newYIndexes{3}] = ...
                                        EmProj.classification_clustering(avgNodeClassSuccess{layer},'-conmat',6);
                            matXLabels{3} = nodeLabels(newXIndexes{3});
                            matYLabels{3} = nodeLabels(newYIndexes{3});                


                        else
                            classificationMat{1} = avgLetterTransClassSuccess{layer};
                            matXLabels{1} = nodeLabels;
                            matYLabels{1} = letterTransLabels;

                            classificationMat{2} = avgNodeTransClassSuccess{layer};
                            matXLabels{2} = nodeLabels;
                            matYLabels{2} = nodeTransLabels;

                            classificationMat{3} = avgNodeClassSuccess{layer};
                            matXLabels{3} = nodeLabels;
                            matYLabels{3} = nodeLabels;                

                        end

                        figure_h1 = EmProj.confusion_matrix(classificationMat{1},classificationLowerLimit,'Node','Letter Transition',...
                                matXLabels{1}, matYLabels{1}, popup);           
                        figure_h2 = EmProj.confusion_matrix(classificationMat{2},classificationLowerLimit,'Node','Node Transition',...
                                matXLabels{2}, matYLabels{2}, popup);
                        figure_h3 = EmProj.confusion_matrix(classificationMat{3},classificationLowerLimit,'Node','Node',...
                                matXLabels{3}, matYLabels{3}, popup);            
                        %regrp = [1 2 5 6 7 8 3 4 15 16 13 14 19 9 10 17 18 11 12 20]
                        if savePlots
                            export_fig(['../','conmat_',groups{grp},'_L',num2str(layer),'_nodesByLetterTrans_',sortPrefix,pngPrefix],'-transparent','-q101','-m3',figure_h1);
                            export_fig(['../','conmat_',groups{grp},'_L',num2str(layer),'_nodesByNodeTrans_',sortPrefix,pngPrefix],'-transparent','-q101','-m3',figure_h2);
                            export_fig(['../','conmat_',groups{grp},'_L',num2str(layer),'_nodesByNodes_',sortPrefix,pngPrefix],'-transparent','-q101','-m3',figure_h3);                
                        end
                        if length(processLayers) > 1 && pauseOn, pause; end
                    end
                case {4,7}
            end
            if ~popup, close all; end
        end%sortMat
    end
%% ------------------------------------------------------------------------
end%classType
end%grp
