%%
%cd('/Volumes/My_HD/Dropbox/matlab/emergentproj');
cd('~/Dropbox/matlab/emergentproj/data');

close all %note, this script currently requires all figures to be closed...

avgSse = zeros(nNets,length(groups));
avgSseStats = zeros(2,length(groups));
for grp = [1 2 3] %[1 2 3]
    clearvars -except grp avgSse avgSseStats
    %--SET THESE--
    if grp < 3
        idList = [1:10]; %loopable
    else
        idList = 11;
    end

    groups = {'network' 'control' 'nullcontrol'};
    groupTitles = {'Trained Networks' 'Control Networks' 'Control Networks (no copy)'};
    %-------------

    nLetterTrans = 20;
    nNodeTrans = 11;
    nNodes = 6;
    
    letterTransCol = 1;
    nodeTransCol = 2;
    nodesCol = 6;
    sseCol = 5;
    
    nNets = length(idList);    
    letterTrans_idxs = cell(1,nLetterTrans);
    nodeTrans_idxs = cell(1,nNodeTrans);
    nodes_idxs = cell(1,nNodes);
    
    avgLetterTransSse = zeros(nNets,nLetterTrans);
    avgNodeTransSse = zeros(nNets,nNodeTrans);
    avgNodesSse = zeros(nNets,nNodes);

    %%
    
    for netId = idList
        dataLabels = importdata([groups{grp},sprintf('%02d',netId),'_trial_labels.txt']);
        dataLabels(:,6) = dataLabels(:,3) + 1; %shift it up by 1 to avoid having 0s
        dataLabels(:,7) = dataLabels(:,4) + 1; %shift it up by 1 to avoid having 0s
    
        for jLetterTrans = 1:nLetterTrans
            letterTrans_idxs{jLetterTrans} = find( dataLabels(:,letterTransCol)==jLetterTrans );
            avgLetterTransSse(netId,jLetterTrans) = mean(dataLabels(letterTrans_idxs{jLetterTrans},sseCol));
        end
        
        for jNodeTrans = 1:nNodeTrans
            nodeTrans_idxs{jNodeTrans} = find( dataLabels(:,nodeTransCol)==jNodeTrans );
            avgNodeTransSse(netId,jNodeTrans) = mean(dataLabels(nodeTrans_idxs{jNodeTrans},sseCol));
        end
        
        for jNodes = 1:nNodes
            nodes_idxs{jNodes} = find( dataLabels(:,nodesCol)==jNodes );
            avgNodesSse(netId,jNodes) = mean(dataLabels(nodes_idxs{jNodes},sseCol));
        end  
        
        avgSse(netId,grp) = mean(dataLabels(:,sseCol));
    end
    %%
    if grp == 3
        seDivisor = sqrt(10);
    else
        seDivisor = sqrt(nNets);
    end
    
    avgLetterTransSseStats(1,:) = sum(avgLetterTransSse,1)/nNets; 
    avgLetterTransSseStats(2,:) = std(avgLetterTransSse)/seDivisor;
    %[sseLetterTransNormfit(1,:), sseLetterTransNormfit(2,:), sseLetterTransNormfitCi, ~] = ...
    %    normfit(avgLetterTransSse,1);
    
    avgNodeTransSseStats(1,:) = sum(avgNodeTransSse,1)/nNets; 
    avgNodeTransSseStats(2,:) = std(avgNodeTransSse)/seDivisor;
    
    avgNodesSseStats(1,:) = sum(avgNodesSse,1)/nNets; 
    avgNodesSseStats(2,:) = std(avgNodesSse)/seDivisor;
    
    avgSseStats(1,grp) = sum(avgSse(:,grp),1)/nNets;
    avgSseStats(2,grp) = std(avgSse(:,grp))/seDivisor;
    
    %% barweb
    % ( barvalues, errors, width, groupnames, 
    %   bw_title, bw_xlabel, bw_ylabel, 
    %   bw_colormap, gridstatus, bw_legend,
    %   error_sides, legend_type)
    colmean = avgLetterTransSseStats(1,:);
    colsd = avgLetterTransSseStats(2,:);
    fig1_h = figure; bar_hs = barweb(colmean,colsd,0.8,[],...
            groupTitles{grp},'Letter Transitions','Average SSE',...
            'gray','y',{'B->T','B->P','S->S','T->S','T->X','S->X','T->T','P->T','X->T','X->V','T->V','P->V','X->X','P->X','X->S','P->S','V->P','V->V','V->E','S->E'},...
            1,'axis',45);
        axes_pos = get(bar_hs.ax,'Position');
        set(bar_hs.ax,'Position',[axes_pos(1) axes_pos(2)+0.1 axes_pos(3)+0.05 axes_pos(4)-0.15])
        current_pos = get(fig1_h,'Position');
        set(fig1_h,'Position',[current_pos(1) current_pos(2) current_pos(3)/1 current_pos(3)/2])
        ylim([0,2]);    
    box on
    
    %%
    colmean = avgNodeTransSseStats(1,:);
    colsd = avgNodeTransSseStats(2,:);
    fig2_h = figure; bar_hs = barweb(colmean,colsd,0.8,[],...
            groupTitles{grp},'Node Transitions','Average SSE',...
            'gray','y',{'1:T (N0)','2:P (N0)','3:S (N1)','4:X (N1)','5:T (N2)','6:V (N2)','7:X (N3)','8:S (N3)','9:P (N4)','10:V (N4)','11:E (N5)'},...
            1,'axis',45);
        axes_pos = get(bar_hs.ax,'Position');
        set(bar_hs.ax,'Position',[axes_pos(1)+.06 axes_pos(2)+0.18 axes_pos(3)+0.01 axes_pos(4)-0.22])
        current_pos = get(fig2_h,'Position');    
        set(fig2_h,'Position',[current_pos(1) current_pos(2) current_pos(3)/2 current_pos(3)/2.5])
        ylim([0,2]);    
    box on
    
    %%

    colmean = avgNodesSseStats(1,:);
    colsd = avgNodesSseStats(2,:);
    fig3_h = figure; bar_hs = barweb(colmean,colsd,0.8,[],...
            groupTitles{grp},'Nodes','Average SSE',...
            'gray','y',{'N0','N1','N2','N3','N4','N5'},...
            1,'axis');
        current_pos = get(fig3_h,'Position');
        set(fig3_h,'Position',[current_pos(1) current_pos(2) current_pos(3)/2 current_pos(3)/2.5])
        ylim([0,2]);
    box on        
        
    %%

    export_fig(['../','sse_',groups{grp},'_letterTrans.png'],'-transparent','-q101','-m2',fig1_h);
    export_fig(['../','sse_',groups{grp},'_nodeTrans.png'],'-transparent','-q101','-m2',fig2_h);
    export_fig(['../','sse_',groups{grp},'_nodes.png'],'-transparent','-q101','-m2',fig3_h);

end%grp

colmean = avgSseStats(1,:);
colsd = avgSseStats(2,:);
fig4_h = figure; bar_hs = barweb(colmean,colsd,0.8,[],...
        groupTitles{grp},'Nodes','Average SSE',...
        'gray','y',{{'Trained'; 'Network'}, {'Untrained';'Control'}, {'Untrained';'Control';'(no copy)'}},...
        1,'axis');
axes_pos = get(bar_hs.ax,'Position');
set(bar_hs.ax,'Position',[axes_pos(1)+0.05 axes_pos(2)+0.1 axes_pos(3) axes_pos(4)-0.15])    
current_pos = get(fig4_h,'Position');
set(fig4_h,'Position',[current_pos(1) current_pos(2) current_pos(3)/2 current_pos(4)/1.5])
ylim([0,2]);
box on
export_fig(['../','sse_overall.png'],'-transparent','-q101','-m2',fig4_h);




