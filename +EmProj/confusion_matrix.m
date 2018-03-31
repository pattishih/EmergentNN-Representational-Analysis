function figure_h = confusion_matrix(plotData, classificationLowerLimit, classifierAxisTitle, comparisonAxisTitle, classifierNames, comparisonNames, popup)
%CONFUSION_MATRIX(PLOTDATA, CLASSIFICATIONAXISTITLE, COMPARISONAXISTITLE, CLASSIFIER NAMES, COMPARISONNAMES)
%  PLOTDATA:  image data

set(0,'DefaultAxesFontName', 'Arial')
set(0,'DefaultTextFontname', 'Arial')

rotationDeg = 40;
%% Create figure
if ~popup
    figure_h = figure('Visible','off');
else
    figure_h = figure;
    set(figure_h,'Units','pixels')
end
    

[m,n] = size(plotData); %largestAxis = max(m,n);
figArea_pos = get(figure_h,'Position');

% Adjust the figure area window

%plotAreaRatio = min(figArea_pos(3),figArea_pos(4))/max(figArea_pos(3),figArea_pos(4));%size ratio of the plot area

if m > n %more rows than columns (i.e., tall)
    if m >= 3*n %if the number of rows is much larger than the number of columns...
        set(figure_h, 'OuterPosition', [figArea_pos(1) figArea_pos(2) figArea_pos(4) figArea_pos(3)])
        
        width = n/m - 0.08; %make the width smaller
        height = 0.8;%

        %min(figArea_pos(3),width)/max(figArea_pos(3),width)
        fracOfPlotFromLeft = ((1-width)/2);%pinch in from plot area left edge
        %min(figArea_pos(4),height)/max(figArea_pos(4),height)
        fracOfPlotFromBottom = ((1-height)/3);%allow max height, while giving room to labels and extras
        %min(figArea_pos(4),height)/max(figArea_pos(4),height)
        fracOfColorbarFromBottom = fracOfPlotFromBottom-0.1;

        xSize = 1-fracOfPlotFromLeft*2;%make sure 2*fracOfPlotFromLeft + xSize roughly equals 1?
        ySize = 1-fracOfPlotFromBottom*3;%make sure 2*fracOfPlotFromBottom + xSize roughly equals 1?

    else
        width = n/m - 0.2; %make the width smaller
        height = 0.55;%

        %min(figArea_pos(3),width)/max(figArea_pos(3),width)
        fracOfPlotFromLeft = ((1-width)/2);%pinch in from plot area left edge
        %min(figArea_pos(4),height)/max(figArea_pos(4),height)
        fracOfPlotFromBottom = ((1-height)/3);%allow max height, while giving room to labels and extras
        %min(figArea_pos(4),height)/max(figArea_pos(4),height)
        fracOfColorbarFromBottom = fracOfPlotFromBottom-0.1;

        xSize = 1-fracOfPlotFromLeft*2;%make sure 2*fracOfPlotFromLeft + xSize roughly equals 1?
        ySize = 1-fracOfPlotFromBottom*2.9;%make sure 2*fracOfPlotFromBottom + xSize roughly equals 1?
    end
elseif m == n %equal
    if m > 6
        set(figure_h, 'OuterPosition', [figArea_pos(1) figArea_pos(2) figArea_pos(3) figArea_pos(3)])

        width = 0.60; %keep the width the same size
        height = 0.8;%

        fracOfPlotFromLeft = ((1-width)/2);%allow max width, while giving room to labels and other horz stuff
        fracOfPlotFromBottom = ((1-height)/1.2);%pinch in from plot area bottom edge
        fracOfColorbarFromBottom = fracOfPlotFromBottom-0.1;

        xSize = 1-fracOfPlotFromLeft*2;
        ySize = 1-fracOfPlotFromBottom*2.3;
    else
        width = 0.60; %keep the width the same size
        height = 0.60;%

        fracOfPlotFromLeft = ((1-width)/2);%allow max width, while giving room to labels and other horz stuff
        fracOfPlotFromBottom = ((1-height)/2);%pinch in from plot area bottom edge
        fracOfColorbarFromBottom = fracOfPlotFromBottom-0.1;

        xSize = 1-fracOfPlotFromLeft*2;
        ySize = 1-fracOfPlotFromBottom*2.5;
    end

else  %or more columns than rows (i.e., wide)
    width = 0.6; %keep the width the same size
    height = m/n - 0.1;%
    
    fracOfPlotFromLeft = ((1-width)/2);%allow max width, while giving room to labels and other horz stuff
    fracOfPlotFromBottom = ((1-height)/3);%pinch in from plot area bottom edge
    fracOfColorbarFromBottom = fracOfPlotFromBottom-0.1;

    xSize = 1-fracOfPlotFromLeft*2;
    ySize = 1-fracOfPlotFromBottom*2.4;
end

%set(figure_h, 'Position', [figArea_pos(1) figArea_pos(2) width height])


%% Create axes
%'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'
%    'FontWeight','bold',...
%    'CameraViewAngleMode','auto',...
%    'DataAspectRatio',[.8 1 1],...
%    'DataAspectRatioMode','auto',...   
%    'PlotBoxAspectRatio',[width height 1],...

[fracOfPlotFromLeft fracOfPlotFromBottom xSize ySize]

axes_h = axes('Parent',figure_h,...
    'PlotBoxAspectRatioMode','manual',...
    'YTickLabel',comparisonNames,...
    'YTick',[1:m],...
    'XTickLabel',classifierNames,...
    'XTick',[1:n],...
    'YDir','reverse',...
    'XAxisLocation','top',...
    'TickDir','in',...
    'LineWidth',1,...
    'TickLength',[0 0],...
    'Position',[fracOfPlotFromLeft fracOfPlotFromBottom xSize ySize],...
    'Layer','top',...
    'FontSize',14,...
    'CLim',[0 1]);



xlim(axes_h,[0.5 n+0.5]);
ylim(axes_h,[0.5 m+0.5]);
box(axes_h,'on');
hold(axes_h,'all');


%% Create image
plotDataLimited = (plotData > classificationLowerLimit).*plotData;


image(plotDataLimited,'Parent',axes_h,'CDataMapping','scaled');

axis image
axes_pos = get(axes_h,'Position');
%pause 

xTickLabels_h = rotateticklabel(axes_h,rotationDeg);
grid_rectangular( 0.5, n+0.5, n+1, 0.5, m+0.5, m+1 );
xlabel ( [classifierAxisTitle,' Classifiers'])
ylabel ( [comparisonAxisTitle,' Classification Accuracy'])
title ('Confusion Matrix', 'FontWeight','bold','FontSize',16)


%% Create colorbar
colorbar_h = colorbar('peer',axes_h,'SouthOutside');
%'FontSize',12);

%box(colorbar_h,'off');
%pause

%colorbar_h = findobj(gcf,'Tag','Colorbar');
colorbar_pos = get(colorbar_h,'Position');


newColorbarXSize = colorbar_pos(3)*0.3;
newColorbarYSize = colorbar_pos(4)*0.5;

set(colorbar_h,...
    'Position',...
    [0.5 - newColorbarXSize/2 ...horizontal pos of bottom-left corner along x
     axes_pos(2)-fracOfColorbarFromBottom ...vertical pos of bottom-left corner along y
     newColorbarXSize ...x size
     newColorbarYSize]); %y size

 hold off
 
 %if ~popup, close all; end
     
%colorbar('YTickLabel',...
%    {'Freezing','Cold','Cool','Neutral',...
%     'Warm','Hot','Burning','Nuclear'})


%axis equal; 
%axis square