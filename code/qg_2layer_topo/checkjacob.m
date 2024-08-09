% % load('QG_DATA_topo10_nu1e-12_beta3_K128_dt5e-4_subs_cg.mat')
% ke_cg = ke;
% ape_cg = ape;
% ene_cg = ene;
% etp_cg = etp;
% 
% load('QG_DATA_topo10_nu1e-12_beta3_K128_dt5e-4_subs.mat')

% Set simulation parameters; ocean regime
N = 128;       % Number of points in each direction
dt = 5E-4;     % initial time step size
Nt = 6*1E4;      % Number of time steps
% qlim = 1E4;  % if any q > qlim, simulation stops
% s_rate = 4;   % subsampling rate
% cut = 48;      % number of truncated modes = cut*2+1
% 
% % Set physical parameters
% kd = 10;       % Nondimensional deformation wavenumber
% kb = sqrt(3); % Nondimensional beta wavenumber, beta = kb^2 
% U = 1;         % zonal shear flow
% r = 9;         % Nondimensional Ekman friction coefficient
% nu = 1E-12;    % Coefficient of biharmonic vorticity diffusion
% H = 10;       % Topography parameter 
% 
% % Set up hyperviscous PV dissipation
% k = [0:N/2 -N/2+1:-1]';  % wavenumbers
% L = zeros([N N 2]);
% for jj=1:N
%     for ii=1:N
%         kr = sqrt(k(ii)^2+k(jj)^2);
%         L(ii,jj,:) = -nu*kr^8;
%     end
% end
% 
% clear kr ii jj
% 
% % Initialize topography 
% dx=2*pi/N;
% [X,Y]=meshgrid(-pi:dx:pi-dx,-pi:dx:pi-dx);
% topo = H * (cos(X)+2*cos(2*Y));
% topo = topo-mean(mean(topo));
% global hk 
% hk = fft2(topo);
% 
% % Initialize
% t = 0;
% 
% % Diagnosticse
% tstart = Nt-40000;
% countDiag = 100; % Compute diagnostics every countDiag steps
% T = zeros(1,Nt/countDiag);
% vb = zeros(1,Nt/countDiag);       % flux transport
% utz = zeros(N,Nt/countDiag);      % zonal mean flow
% ke = zeros(N/2+1,Nt/countDiag);   % kinetic energy
% ape = zeros(N/2+1,Nt/countDiag);  % available potential energy
% ene = zeros(N/2+1,Nt/countDiag);   % energy
% etp = zeros(N/2+1,Nt/countDiag);  % enstrophy energy
% ke_cg = zeros(N/2+1,Nt/countDiag);   % kinetic energy
% ape_cg = zeros(N/2+1,Nt/countDiag);  % available potential energy
% ene_cg = zeros(N/2+1,Nt/countDiag);   % energy
% etp_cg = zeros(N/2+1,Nt/countDiag);  % enstrophy energy
% 
% for ii=1:Nt
%     if mod(ii,countDiag)==0
%         % if any(isnan(q(:))), break, end
%         T(ii/countDiag)=t;
%         [KE,APE,E,ETP] = Spectrum_topo(jacob_t(:,:,ii/countDiag,:),params);
%         ke(:,ii/countDiag) = KE; ape(:,ii/countDiag) = APE; ene(:,ii/countDiag) = E; etp(:,ii/countDiag) = ETP;
%         [KE,APE,E,ETP] = Spectrum_topo(jacob_cg_t(:,:,ii/countDiag,:),params);
%         ke_cg(:,ii/countDiag) = KE; ape_cg(:,ii/countDiag) = APE; ene_cg(:,ii/countDiag) = E; etp_cg(:,ii/countDiag) = ETP;
% 
%         display(['iteration i = ', num2str(ii), '; time step dt = ',num2str(dt), ', ene = ',num2str(sum(KE+APE))]);
%         toc;
%     end
% end

h1 = figure(2);
set(h1, 'Position', [20, 20, 500, 300]); % Set the figure size ([left, bottom, width, height])
subplot(1,2,1)
loglog([0:N/2],mean(ke(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
loglog([0:N/2],mean(ke_cg(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
legend('KE', 'KE\_CG', 'Location', 'best'); % Add a legend
title('kinetic energy spectrum'); xlabel('wavenumber');
subplot(1,2,2)
loglog([0:N/2],mean(ape(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
loglog([0:N/2],mean(ape_cg(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
title('potential energy spectrum'); xlabel('wavenumber');
legend('APE', 'APE\_CG', 'Location', 'best'); % Add a legend
print(h1, 'energy_beta3_jacobdiff.png', '-dpng', '-r150')

h3 = figure(3);
set(h3, 'Position', [20, 20, 500, 300]); % Set the figure size ([left, bottom, width, height])
subplot(1,2,1)
loglog([0:N/2],mean(ene(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
loglog([0:N/2],mean(ene_cg(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
legend('ENE', 'ENE\_CG', 'Location', 'best'); % Add a legend
title('energy spectrum'); xlabel('wavenumber');
subplot(1,2,2)
loglog([0:N/2],mean(etp(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
loglog([0:N/2],mean(etp_cg(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
legend('ETP', 'ETP\_CG', 'Location', 'best'); % Add a legend
title('enstrophy spectrum'); xlabel('wavenumber');
print(h3, 'mode_ene_ens_beta3_jacobdiff.png', '-dpng', '-r150')

% h4 = figure(4);
% % Preparing the data
% K=128;
% mean_jacob = mean(jacob_t, 3);
% mean_jacob_cg = mean(jacob_cg_t, 3);
% dataplot = zeros(K, K, 1, 4);
% diff2 = mean_jacob_cg(:,:,:,2) - mean_jacob(:,:,:,2);
% dataplot(:,:,1,1) = real(diff2);
% dataplot(:,:,1,2) = imag(diff2);
% dataplot(:,:,1,3) = real(mean_jacob(:,:,:,2));
% dataplot(:,:,1,4) = imag(mean_jacob(:,:,:,2));
% 
% % Rolling the array for centering
% dataplot = circshift(dataplot, [K/2, K/2, 0, 0]);
% 
% % Frequency vector
% Kx = [-N/2+1:N/2];
% 
% % Plotting
% r_cut = 64;
% i=1;
%     for j = 1:4
%         n_tick = K/r_cut*2+1;
%         maxabs = max(max(abs(dataplot(:,:,i,j))));
%         subplot(1, 4, (i-1)*4 + j);
%         imagesc(Kx, Kx, dataplot(:,:,i,j), [-maxabs maxabs]);
%         colormap('jet');
%         colorbar;
%         axis square;
%         xticks(linspace(-K/2, K/2, n_tick));
%         yticks(linspace(-K/2, K/2, n_tick));
%         xlim([-r_cut r_cut]);
%         ylim([-r_cut r_cut]);
%         title(sprintf('Subplot %d, %d', i, j));
%     end
% 
% % Titles for each subplot
% subplot_titles = {'$Diff\ Re(Jacob_2)$', '$Diff\ Im(Jacob_2)$', '$Re(Jacob_2)$', '$Im(Jacob_2)$'};
% for k = 1:4
%     subplot(1,4,k);
%     title(subplot_titles{k}, 'Interpreter', 'latex');
% end
% 
% % Adjust layout and save the figure
% set(gcf, 'Position', [100, 100, 1000, 200]); % Modify as needed
% print(h4, 'mode_tmean_jacobdiff.png', '-dpng', '-r150')
