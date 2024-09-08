ht = figure(12);
N = 128;       % Number of points in each direction
dx=2*pi/N;
[X,Y]=meshgrid(-pi:dx:pi-dx,-pi:dx:pi-dx);
mu = [1, 1.5];
sigma = [.2 0.0; 0.0 .2];
F = mvnpdf([X(:) Y(:)], mu, sigma);
F = reshape(F, size(X));
topo = H * (cos(X)+2*cos(2*Y) + 4*F);
topo = topo-mean(mean(topo));
[xx,yy] = meshgrid(linspace(-pi,pi,params.N));
set(ht, 'Position', [20, 20, 280, 250]); % Set the figure size ([left, bottom, width, height])
contour(xx,yy,topo,50); 
% caxis([-250 250]);
colorbar;
title('topography');
xlabel('x'); ylabel('y');
% print(ht, 'topo3.png', '-dpng', '-r150') 
