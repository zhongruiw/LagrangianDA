% This script solves nondimensional 2-layer QG with equal layers and a
% rigid lid in a doubly-periodic domain.
rng(2024);

% Set simulation parameters; ocean regime
N = 128;       % Number of points in each direction
dt = 2E-3;     % initial time step size
Nt = 205*1E3;      % Number of time steps
qlim = 1E4;  % if any q > qlim, simulation stops
s_rate = 8;   % subsampling rate
cut = 48;      % number of truncated modes = cut*2+1

% Set physical parameters
kd = 10;       % Nondimensional deformation wavenumber
kb = sqrt(111); % Nondimensional beta wavenumber, beta = kb^2 
U = 1;         % zonal shear flow
r = 9;         % Nondimensional Ekman friction coefficient
nu = 1e-12;    % Coefficient of biharmonic vorticity diffusion
H = 40;       % Topography parameter 

% Set up hyperviscous PV dissipation
k = [0:N/2 -N/2+1:-1]';  % wavenumbers
L = zeros([N N 2]);
for jj=1:N
    for ii=1:N
        kr = sqrt(k(ii)^2+k(jj)^2);
        L(ii,jj,:) = -nu*kr^8;
    end
end

clear kr ii jj

% Put useful stuff into a struct
params = struct('U',U, 'kd',kd, 'kb',kb, 'r',r, 'nu',nu, 'N',N, 'dt',dt, 'H',H);

% Initialize
t = 0;
% Initialize topography 
dx=2*pi/N;
[X,Y]=meshgrid(-pi:dx:pi-dx,-pi:dx:pi-dx);
% mu = [1, 1.5];
% sigma = [.2 0.0; 0.0 .2];
% F = mvnpdf([X(:) Y(:)], mu, sigma);
% F = reshape(F, size(X));
F = 0;
topo = H * (cos(X)+2*cos(2*Y) + 4*F);
topo = topo-mean(mean(topo));
global hk 
hk = fft2(topo);
% initialize potential vorticity
qp(:,:,2) = 10*randn(params.N); % qp is actually the 'relative PV', without include topograpy
qp(:,:,2) = qp(:,:,2)-mean(mean(qp(:,:,2)));
qp(:,:,1) = qp(:,:,2);
q = fft2(qp);
[xx,yy] = meshgrid(linspace(-pi,pi,params.N));
% initialize shear flow
Ut = params.U;

% Diagnosticse
tstart = Nt-200000;
countDiag = 100; % Compute diagnostics every countDiag steps
T = zeros(1,Nt/countDiag);
vb = zeros(1,Nt/countDiag);       % flux transport
utz = zeros(N,Nt/countDiag);      % zonal mean flow
ke = zeros(N/2+1,Nt/countDiag);   % kinetic energy
ape = zeros(N/2+1,Nt/countDiag);  % available potential energy
ene = zeros(N/2+1,Nt/countDiag);   % energy
etp = zeros(N/2+1,Nt/countDiag);  % enstrophy energy
ene_t = zeros(1,Nt/countDiag);   % energy
etp_t = zeros(1,Nt/countDiag);  % enstrophy energy
% psi_1_t = zeros(N-cut*2-1,N-cut*2-1,(Nt-tstart)/s_rate);  % upper layer stream function
% psi_2_t = zeros(N-cut*2-1,N-cut*2-1,(Nt-tstart)/s_rate);  % lower layer stream function
psi_1_t = zeros(N,N,(Nt-tstart)/s_rate);  % upper layer stream function
psi_2_t = zeros(N,N,(Nt-tstart)/s_rate);  % lower layer stream function
psi_1_t_fine = zeros(N-cut*2-1,N-cut*2-1,(Nt-tstart));
% v1_t = zeros(N,N,(Nt-tstart)/countDiag);  % upper layer stream function
% u1_t = zeros(N,N,(Nt-tstart)/countDiag);  % upper layer stream function
qp_t = zeros(N,N,2,9);
% jacob_t = zeros(N,N,(Nt-tstart)/s_rate,2); % save Jacobian 
% jacob_cg_t = zeros(N,N,(Nt-tstart)/s_rate,2); % save Jacobian 
% rhs_q_t = zeros(N,N,(Nt-tstart)/s_rate,2); % save rhs 
% rhs_q21_t = zeros(N,N,(Nt-tstart)/s_rate); % save rhs 
% rhs_q22_t = zeros(N,N,(Nt-tstart)/s_rate); % save rhs 
% rhs_q23_t = zeros(N,N,(Nt-tstart)/s_rate); % save rhs 

% mode truncation
cutRow = (N/2-cut+1):(N/2+cut+1);

% adaptive stepping stuff:
tol= 1E-1;
r0 = .8*tol;
params
% Main loop 
tic;

dX = 1i*repmat(k',[N 1 2]);
dY = 1i*repmat(k,[1 N 2]);
Laplacian = dX(:,:,1).^2+dY(:,:,1).^2;
InvBT = 1./Laplacian; InvBT(1,1) = 0;
InvBC = 1./(Laplacian-kd^2);InvBC(1,1) = 0;

qp_t(:,:,:,1) = qp;
for ii=1:Nt
    if mod(ii,countDiag)==0
        if any(isnan(q(:))), break, end
        T(ii/countDiag)=t;
        [KE,APE,E,ETP] = Spectrum_topo(q,params);
        ke(:,ii/countDiag) = KE; ape(:,ii/countDiag) = APE; ene(:,ii/countDiag) = E; etp(:,ii/countDiag) = ETP;
        [VB,UTZ,Et,ETPt] = QG_Diagnostics_topo(q,params);
        vb(ii/countDiag) = VB; utz(:,ii/countDiag) = UTZ; ene_t(ii/countDiag) = Et; etp_t(ii/countDiag)=ETPt; 

        display(['iteration i = ', num2str(ii), '; time step dt = ',num2str(dt), ', ene = ',num2str(sum(KE+APE))]);
        toc;
    end

    % % Euler
    % [k0,psik,jacob0] = RHS_Spectral_topo(q,params,Ut);
    % l0 = L.*q;
    % % Successful step, proceed to evaluation
    % t = t+dt;
    % qp = real(ifft2(q+dt*(k0+l0)));
    % q = fft2(qp);

    M = 1./(1-.25*dt*L);
    % First stage ARK4
    [k0,psik] = RHS_Spectral_topo(q,params,Ut);
    l0 = L.*q;
    % u0 = RHS_meanFlow_topo(psik,params); 
    % Second stage
    q1 = M.*(q+.5*dt*k0+.25*dt*l0);
    % U1 = Ut+.5*dt*u0;    
    [k1,psik] = RHS_Spectral_topo(q1,params,Ut); 
    l1 = L.*q1;
    % u1 = RHS_meanFlow_topo(psik,params); 
    % Third stage
    q2 = M.*(q+dt*(13861*k0/62500+6889*k1/62500+8611*l0/62500-1743*l1/31250));
    % U2 = Ut + dt*(13861*u0/62500+6889*u1/62500); 
    [k2,psik] = RHS_Spectral_topo(q2,params,Ut); 
    l2 = L.*q2;
    % u2 = RHS_meanFlow_topo(psik,params); 
    % Fourth stage
    q3 = M.*(q+dt*(-0.04884659515311858*k0-0.1777206523264010*k1+0.8465672474795196*k2...
    +0.1446368660269822*l0-0.2239319076133447*l1+0.4492950415863626*l2));
    % U3 = Ut + dt*(-0.04884659515311858*u0-0.1777206523264010*u1+0.8465672474795196*u2); 
    [k3,psik] = RHS_Spectral_topo(q3,params,Ut); 
    l3 = L.*q3;
    % u3 = RHS_meanFlow_topo(psik,params); 
    % Fifth stage
    q4 = M.*(q+dt*(-0.1554168584249155*k0-0.3567050098221991*k1+1.058725879868443*k2...
    +0.3033959883786719*k3+0.09825878328356477*l0-0.5915442428196704*l1...
    +0.8101210538282996*l2+0.2831644057078060*l3));
    % U4 = Ut + dt*(-0.1554168584249155*u0-0.3567050098221991*u1+1.058725879868443*u2...
    % +0.3033959883786719*u3); 
    [k4,psik] = RHS_Spectral_topo(q4,params,Ut); 
    l4 = L.*q4;
    % u4 = RHS_meanFlow_topo(psik,params); 
    % Sixth stage
    q5 = M.*(q+dt*(0.2014243506726763*k0+0.008742057842904184*k1+0.1599399570716811*k2...
    +0.4038290605220775*k3+0.2260645738906608*k4+0.1579162951616714*l0...
    +0.1867589405240008*l2+0.6805652953093346*l3-0.2752405309950067*l4));
    % U5 = Ut + dt*(0.2014243506726763*u0+0.008742057842904184*u1+0.1599399570716811*u2...
    % +0.4038290605220775*u3+0.2260645738906608*u4); 
    [k5,psik] = RHS_Spectral_topo(q5,params,Ut); 
    l5 = L.*q5;
    % u5 = RHS_meanFlow_topo(psik,params); 

    % % Error control
    %  r1 = dt*max(max(max(abs(ifft2(0.003204494398459*(k0+l0) -0.002446251136679*(k2+l2)-0.021480075919587*(k3+l3)...
    %      +0.043946868068572*(k4+l4) -0.023225035410765*(k5+l5))))));
    %  if r1>tol,dt=.75*dt;continue,end

    % Successful step, proceed to evaluation
    t = t+dt;
    qp = real(ifft2(q+dt*(0.1579162951616714*(k0+l0)+0.1867589405240008*(k2+l2)+...
    0.6805652953093346*(k3+l3)-0.2752405309950067*(k4+l4)+(k5+l5)/4)));
    q = fft2(qp);
    % Ut = Ut+dt*(0.1579162951616714*(u0)+0.1867589405240008*(u2)+...
    % 0.6805652953093346*(u3)-0.2752405309950067*(u4)+(u5)/4); 

    % % step size adjustment: EPS, PI.3.4 ; divide by 4 for a 4th order
    % % method with 3rd order embedded
    %  dt = ((.75*tol/r1)^(.3/4))*((r0/r1)^(.4/4))*dt;
    %  r0=r1;
    if any(abs(qp(:))>qlim)
        fprintf(['qp = ', num2str(max(abs(qp(:)))),'\n']);
        break
    end

    if mod(ii,1000)==0 && ii/1000+1 <=9
        qp_t(:,:,:,ii/1000+1) = qp;
    end

    if ii > tstart
        % Invert for psi
        q_bt = .5*(q(:,:,1) + q(:,:,2));
        q_bc = .5*(q(:,:,1) - q(:,:,2));
        psi_bt = InvBT.*(q_bt - 0.5*hk(:,:)); 
        psi_bc = InvBC.*(q_bc + 0.5*hk(:,:)); 
        temp1 = psi_bt+psi_bc;
        temp1_ = temp1;
        temp1_(cutRow,:) = [];
        temp1_(:,cutRow) = [];
        psi_1_t_fine(:,:,(ii-tstart)) = temp1_;
        if mod((ii-tstart),s_rate)==0
            temp2 = psi_bt-psi_bc;
            % temp2(cutRow,:) = [];
            % temp2(:,cutRow) = [];
            psi_1_t(:,:,(ii-tstart)/s_rate) = temp1;
            psi_2_t(:,:,(ii-tstart)/s_rate) = temp2;

            % save Jacobian
        %     jacob_t(:,:,(ii-tstart)/s_rate,:) = -dt*(0.1579162951616714*(jacob0)+0.1867589405240008*(jacob2)+...
        % 0.6805652953093346*(jacob3)-0.2752405309950067*(jacob4)+(jacob5)/4);
        %     jacob_cg_t(:,:,(ii-tstart)/s_rate,:) = -dt*(0.1579162951616714*(jacob_cg0)+0.1867589405240008*(jacob_cg2)+...
        % 0.6805652953093346*(jacob_cg3)-0.2752405309950067*(jacob_cg4)+(jacob_cg5)/4);
        %     rhs_q_t(:,:,(ii-tstart)/s_rate,:) = dt*(0.1579162951616714*(k0+l0)+0.1867589405240008*(k2+l2)+...
        % 0.6805652953093346*(k3+l3)-0.2752405309950067*(k4+l4)+(k5+l5)/4);
        %     rhs_q21_t(:,:,(ii-tstart)/s_rate) = dt*(0.1579162951616714*(rhs_2_1_0)+0.1867589405240008*(rhs_2_1_2)+...
        % 0.6805652953093346*(rhs_2_1_3)-0.2752405309950067*(rhs_2_1_4)+(rhs_2_1_5)/4);
        %     rhs_q22_t(:,:,(ii-tstart)/s_rate) = dt*(0.1579162951616714*(rhs_2_2_0)+0.1867589405240008*(rhs_2_2_2)+...
        % 0.6805652953093346*(rhs_2_2_3)-0.2752405309950067*(rhs_2_2_4)+(rhs_2_2_5)/4);
        %     rhs_q23_t(:,:,(ii-tstart)/s_rate) = dt*(0.1579162951616714*(rhs_2_3_0)+0.1867589405240008*(rhs_2_3_2)+...
        % 0.6805652953093346*(rhs_2_3_3)-0.2752405309950067*(rhs_2_3_4)+(rhs_2_3_5)/4);
            % % jacob_t(:,:,(ii-tstart)/s_rate,:) = -dt*jacob0;
            % % rhs_q_t(:,:,(ii-tstart)/s_rate,:) = dt*(k0+l0);
        end
    end
        % if mod(ii-tstart,countDiag)==0
        %     % Real-Space quantities
        %     v1 = real(ifft2(-dX(:,:,1).*psi_1_t(:,:,ii-tstart)));
        %     u1 = real(ifft2(dY(:,:,1).*psi_1_t(:,:,ii-tstart)));
        %     v1_t(:,:,(ii-tstart)/countDiag) = v1;
        %     u1_t(:,:,(ii-tstart)/countDiag) = u1;        
        % end
end
toc

% shift the domain from [0,2pi) to [-pi,pi)
qp = circshift(qp, [N/2, 0]);
qp = circshift(qp, [0, N/2]);
qp_t = circshift(qp_t, [N/2, 0, 0, 0]);
qp_t = circshift(qp_t, [0, N/2, 0, 0]);

if any(isnan(q(:)))
    fprintf('NaN\n')
else
    save('QG_DATA_topo40_nu1e-12_beta111_K128_dt2e-3_subs.mat','ii','countDiag','dt','tol','params','T','ke','ape','ene','etp','vb','utz', 'qp','topo','psi_1_t','psi_2_t','s_rate','cut','psi_1_t_fine','-v7.3'); 
end

h5 = figure(5);
for i = 1:9
    t1 =(i-1) * 1000  * dt;
    subplot(3,3,i)
    hold on
    contour(xx,yy,qp_t(:,:,1,i),200); 
    caxis([-250 250]);
    colorbar;
    title(['upper layer t=', num2str(t1)]);
    xlabel('x'); ylabel('y');
    % set(gca,'fontsize',12)
end
print(h5, 'upper_topo40_nu1e-12_beta111_K128_dt2e-3_snap.png', '-dpng', '-r150') 


h6 = figure(6);
for i = 1:9
    t1 =(i-1) * 1000  * dt;
    subplot(3,3,i)
    hold on
    contour(xx,yy,qp_t(:,:,2,i),200); 
    caxis([-250 250]);
    colorbar;
    title(['lower layer t=', num2str(t1)]);
    xlabel('x'); ylabel('y');
    % set(gca,'fontsize',12)
end
print(h6, 'lower_topo40_nu1e-12_beta111_K128_dt2e-3_snap.png', '-dpng', '-r150') 

% t=20;
% t=10;
h = figure(1);
set(h, 'Position', [20, 20, 600, 210]); % Set the figure size ([left, bottom, width, height])
subplot(1,2,1)
contour(xx,yy,qp(:,:,1),200); 
caxis([-250 250]);
colorbar;
title(['upper layer mode at t = ', num2str(t)]);
xlabel('x'); ylabel('y');
subplot(1,2,2)
contour(xx,yy,qp(:,:,2),200); 
caxis([-250 250]);
colorbar;  % need ot add topography here if 'relative PV' in integration is used
title(['lower layer mode at t = ', num2str(t)]);
xlabel('x'); ylabel('y');
print(h, 'mode_topo40_nu1e-12_beta111_K128_dt2e-3.png', '-dpng', '-r150') 

h1 = figure(2);
set(h1, 'Position', [20, 20, 500, 300]); % Set the figure size ([left, bottom, width, height])
subplot(1,2,1)
loglog([0:N/2],mean(ke(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
title('kinetic energy spectrum'); xlabel('wavenumber');
subplot(1,2,2)
loglog([0:N/2],mean(ape(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
title('potential energy spectrum'); xlabel('wavenumber');
print(h1, 'energy_topo40_nu1e-12_beta111_K128_dt2e-3.png', '-dpng', '-r150')

h3 = figure(3);
set(h3, 'Position', [20, 20, 500, 300]); % Set the figure size ([left, bottom, width, height])
subplot(1,2,1)
loglog([0:N/2],mean(ene(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
title('energy spectrum'); xlabel('wavenumber');
subplot(1,2,2)
loglog([0:N/2],mean(etp(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
title('enstrophy spectrum'); xlabel('wavenumber');
print(h3, 'mode_ene_ens_topo40_nu1e-12_beta111_K128_dt2e-3.png', '-dpng', '-r150')

h4 = figure(4);
set(h4, 'Position', [20, 20, 500, 300]); % Set the figure size ([left, bottom, width, height])
subplot(1,2,1)
plot([1:size(ene_t,2)],ene_t,'.-', 'LineWidth',1); hold on;
title('energy series'); xlabel('time');
subplot(1,2,2)
plot([1:size(etp_t,2)],etp_t,'.-', 'LineWidth',1); hold on;
title('enstrophy series'); xlabel('time');
print(h4, 'series_ene_ens_topo40_nu1e-12_beta111_K128_dt2e-3.png', '-dpng', '-r150')