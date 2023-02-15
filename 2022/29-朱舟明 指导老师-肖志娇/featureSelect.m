%特征选择
function [idx,KN_point,weights] = featureSelect(dataX,dataY,dataName)
    featureNum = size(dataX,2);
    [idx,weights] = relieff(dataX,dataY,10);%数据，数据标签，10个最邻近样本
    weights(isnan(weights))=0;
    point = [1:featureNum;sort(weights,'des')];
    dis = zeros(1,featureNum);
    for i = 1:featureNum %求曲线上的每个点到直线的距离
        dis(i) = abs(det([point(:,featureNum)-point(:,1),point(:,i)-point(:,1)]))/norm(point(:,featureNum)-point(:,1));
    end
    [~,KN_point] = max(dis);

    figure;
    xlabel('featureNum');
    ylabel('weights');
    title('拐点图');
    hold on;
    plot(point(1,:),point(2,:),'b');
    hold on;
    plot([1,featureNum],[max(weights),min(weights)],'r--');
    hold on;
    text(KN_point,point(2,KN_point),['  (' num2str(KN_point) ',' num2str(point(2,KN_point)) ')'])
    plot(KN_point,point(2,KN_point),'r*');%拐点，红色的*
    img =gcf;
    print(img, '-dpng', '-r600', ['result/' dataName '拐点图.png']);


end

