color1 = '#125B50';
color2 = '#FF6363';
color3 = '#F8B400';
color4 = '#8FBDD3';
color5 = '#876445';

x = [10, 15, 20, 25, 30, 35, 40, 45, 50];
y = [92.3056	90.39	92.5	91.39	92.61	92.08	91.27	90.58	90.72];
yy = [92.3056	92.03	92.3056	92.3056	92.4167	92.0278	91.89	90.83	90.03];

figure
plot(x, y,'-o','color',color1,'LineWidth',1.5);hold on;
plot(x, yy,'-o','color',color2,'LineWidth',1.5);hold on;
xlabel('Rate of selected features (%)');
ylabel('ACC (%)');hold on;
legend('unsupervised','supervised');

figure
contour(flip(W));hold on;
xlabel('Matrix dimension of column');
ylabel('Matrix dimension of row');hold on;

y = tsne([test_data_v1;test_data_v2;test_data_v3]');
figure
gscatter(y(:, 1), y(:, 2), test_label, [], [], 20);

yy = tsne([test_data_v1;test_data_v2;test_data_v3]' * W);
figure
gscatter(yy(:, 1), yy(:, 2), test_label, [], [], 20);