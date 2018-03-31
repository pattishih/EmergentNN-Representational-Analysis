%%
cd /Volumes/My_HD/Dropbox/matlab/emergentproj/data


grpList = [1];%loopable [1 2 3]: 1(trained); 2(control); 3(control II)


for grp = grpList
    if grp < 3
        idList = 1:10;
        seDivisor = sqrt(length(idList));
    else
        idList = 11;
        seDivisor = sqrt(10);
    end
    groups = {'network' 'control' 'nullcontrol'};
    
    for netId = idList
        epochdata{netId} = importdata([groups{grp},sprintf('%02d',netId),'_epoch.txt']);
        epochSse(:,netId) = epochdata{netId}(:,3);
    end
    
    avgEpochSseStats = mean(epochSse,2);
    se2EpochSseStats = (std(epochSse,0,2)/seDivisor)*2;
    x = linspace(1,200,200);
    fig_h = figure;
    fig_pos = get(fig_h,'Position');
    set(fig_h,'Position',[fig_pos(1) fig_pos(2) fig_pos(3)*.45 fig_pos(4)*.75])
    plot(x,avgEpochSseStats,'k-','LineWidth',2)
    hold on
    plot(x,avgEpochSseStats+se2EpochSseStats,'k:')
    plot(x,avgEpochSseStats-se2EpochSseStats,'k:')
    ylim([0 1])
    xlabel('Epochs','FontSize',14)
    ylabel('Average SSE','FontSize',14)
    hold off
    export_fig('../sse_epochStats.pdf','-transparent')
end