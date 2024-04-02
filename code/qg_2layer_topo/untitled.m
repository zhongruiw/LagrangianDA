% u1 = real(ifft2(dY(:,:,1).*psi_1_t(:,:,100)));
% qps = circshift(qp, [N/2, 0]);
% qps = circshift(qps, [0, N/2]);
% X = xx(1,:);
h = figure(333);
set(h, 'Position', [20, 20, 600, 250]); % Set the figure size ([left, bottom, width, height])
subplot(1,2,1)
plot(mean(qp(:,:,1),1), 'LineWidth', 2); 
hold on
% plot(topo(1,:));
title('upper layer mean PV of y direction at equilibrium');
xlabel('x');
subplot(1,2,2)
plot(mean(qp(:,:,2),1), 'LineWidth', 2); 
hold on
% plot(topo(1,:));
title('lower layer mean PV of y direction at equilibrium');
xlabel('x');
print(h, 'mean_y_nu4e-14_beta25_equil.png', '-dpng', '-r150')

% figure(333);
% contour(xx,yy,topo,50); 
% % caxis([-250 250]);
% % colorbar;s
% title('topology');
% xlabel('x'); ylabel('y');

h = figure(222);
set(h, 'Position', [20, 20, 600, 250]); % Adjust the figure size for better visibility

timeIndices = [1:9] * 1000 * dt; % Example time indices, adjust as necessary
X = 1:size(qp_t,2); % Assuming the spatial dimension is the same for all layers and times

% For the upper layer
subplot(1,2,1);
% Prepare data for visualization
upperLayerData = squeeze(mean(qp_t(:,:,1,:), 1)); % Squeeze to remove singleton dimensions
% Flip the y-axis data (time) to have the earliest time at the bottom
imagesc(X, timeIndices, upperLayerData');
set(gca, 'YDir', 'normal'); % Correct the y-axis direction
xlabel('X');
ylabel('Time');
title('Upper Layer Mean PV of Y Direction');
colorbar; % Add a color bar to indicate the value of qp_t

% For the lower layer
subplot(1,2,2);
% Prepare data for visualization
lowerLayerData = squeeze(mean(qp_t(:,:,2,:), 1)); % Squeeze to remove singleton dimensions
% Flip the y-axis data (time) to have the earliest time at the bottom
imagesc(X, timeIndices, lowerLayerData');
set(gca, 'YDir', 'normal'); % Correct the y-axis direction
xlabel('X');
ylabel('Time');
title('Lower Layer Mean PV of Y Direction');
colorbar; % Add a color bar to indicate the value of qp_t

% Adjust the figure and subplots layout
set(gcf, 'Position', [100, 100, 600, 250]); % Optionally resize figure to make everything fit nicely
print(h, 'mean_y_nu4e-14_beta25_snap.png', '-dpng', '-r150')
