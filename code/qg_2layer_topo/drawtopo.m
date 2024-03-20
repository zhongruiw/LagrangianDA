ht = figure(12);
set(ht, 'Position', [20, 20, 280, 250]); % Set the figure size ([left, bottom, width, height])
contour(xx,yy,topo,50); 
% caxis([-250 250]);
colorbar;
title('topography');
xlabel('x'); ylabel('y');
print(ht, 'topo075.png', '-dpng', '-r150') 
