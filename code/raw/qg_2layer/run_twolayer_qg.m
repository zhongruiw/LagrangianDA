% This script solves nondimensional 2-layer QG with equal layers and a
% rigid lid in a doubly-periodic domain.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set simulation parameters; ocean regime
N = 128;       % Number of points in each direction
dt = 1E-6;     % initial time step size
Nt = 100*1E5;      % Number of time steps
qlim = 1E4;  % if any q > qlim, simulation stops

% Set physical parameters
kd = 10;       % Nondimensional deformation wavenumber
kb = sqrt(111);        % Nondimensional beta wavenumber, beta = kb^2 
U = 1;         % zonal shear flow
r = 8;         % Nondimensional Ekman friction coefficient
nu = 5E-15;    % Coefficient of biharmonic vorticity diffusion


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
params = struct('U',U, 'kd',kd, 'kb',kb, 'r',r, 'nu',nu, 'N',N, 'dt',dt);

% Initialize
t = 0;
qp(:,:,2) = 10*randn(params.N);
qp(:,:,2) = qp(:,:,2)-mean(mean(qp(:,:,2)));
qp(:,:,1) = qp(:,:,2);
q = fft2(qp);
[xx,yy] = meshgrid(linspace(-pi,pi,params.N));

% Diagnostics
countDiag = 100; % Compute diagnostics every countDiag steps
T = zeros(1,Nt/countDiag);
vb = zeros(1,Nt/countDiag);       % flux transport
utz = zeros(N,Nt/countDiag);      % zonal mean flow
ke = zeros(N/2+1,Nt/countDiag);   % kinetic energy
ape = zeros(N/2+1,Nt/countDiag);  % available potential energy


% adaptive stepping stuff:
tol= 1E-1;
r0 = .8*tol;
params
% Main loop 
tic;
for ii=1:Nt
    if mod(ii,countDiag)==0
        if any(isnan(q(:))), break, end
        T(ii/countDiag)=t;
        [KE,APE] = Spectrum(q,params);
        ke(:,ii/countDiag) = KE; ape(:,ii/countDiag) = APE;
        [VB,UTZ] = QG_Diagnostics(q,params);
        vb(ii/countDiag) = VB; utz(:,ii/countDiag) = UTZ;

        display(['iteration i = ', num2str(ii), '; time step dt = ',num2str(dt), ', ene = ',num2str(sum(KE+APE))]);
        toc;
    end
    M = 1./(1-.25*dt*L);
    % First stage ARK4
    k0 = RHS_Spectral(q,params);
    l0 = L.*q;
    % Second stage
    q1 = M.*(q+.5*dt*k0+.25*dt*l0);
    k1 = RHS_Spectral(q1,params);
    l1 = L.*q1;
    % Third stage
    q2 = M.*(q+dt*(13861*k0/62500+6889*k1/62500+8611*l0/62500-1743*l1/31250));
    k2 = RHS_Spectral(q2,params);
    l2 = L.*q2;
    % Fourth stage
    q3 = M.*(q+dt*(-0.04884659515311858*k0-0.1777206523264010*k1+0.8465672474795196*k2...
    +0.1446368660269822*l0-0.2239319076133447*l1+0.4492950415863626*l2));
    k3 = RHS_Spectral(q3,params);
    l3 = L.*q3;
    % Fifth stage
    q4 = M.*(q+dt*(-0.1554168584249155*k0-0.3567050098221991*k1+1.058725879868443*k2...
    +0.3033959883786719*k3+0.09825878328356477*l0-0.5915442428196704*l1...
    +0.8101210538282996*l2+0.2831644057078060*l3));
    k4 = RHS_Spectral(q4,params);
    l4 = L.*q4;
    % Sixth stage
    q5 = M.*(q+dt*(0.2014243506726763*k0+0.008742057842904184*k1+0.1599399570716811*k2...
    +0.4038290605220775*k3+0.2260645738906608*k4+0.1579162951616714*l0...
    +0.1867589405240008*l2+0.6805652953093346*l3-0.2752405309950067*l4));
    k5 = RHS_Spectral(q5,params);
    l5 = L.*q5;
    % Error control
     r1 = dt*max(max(max(abs(ifft2(0.003204494398459*(k0+l0) -0.002446251136679*(k2+l2)-0.021480075919587*(k3+l3)...
         +0.043946868068572*(k4+l4) -0.023225035410765*(k5+l5))))));
     if r1>tol,dt=.75*dt;continue,end



    % Successful step, proceed to evaluation
    t = t+dt;
    qp = real(ifft2(q+dt*(0.1579162951616714*(k0+l0)+0.1867589405240008*(k2+l2)+...
    0.6805652953093346*(k3+l3)-0.2752405309950067*(k4+l4)+(k5+l5)/4)));
    q = fft2(qp);
    % % step size adjustment: EPS, PI.3.4 ; divide by 4 for a 4th order
    % % method with 3rd order embedded
    %  dt = ((.75*tol/r1)^(.3/4))*((r0/r1)^(.4/4))*dt;
    %  r0=r1;
    if any(abs(qp(:))>qlim)
        fprintf(['qp = ', num2str(max(abs(qp(:)))),'\n']);
        break
    end
end
% if any(isnan(q(:)))
%     fprintf('NaN\n')
% else
%     save('QG_DATA.mat','ii','countDiag','dt','tol','params','T','ke','ape','vb','utz', 'qp');
% end

h = figure;
subplot(1,2,1)
contour(xx,yy,qp(:,:,1),50); colorbar;
title(['barotropic mode at t = ', num2str(t)]);
xlabel('x'); ylabel('y');
subplot(1,2,2)
contour(xx,yy,qp(:,:,2),50); colorbar;
title(['baroclinic mode at t = ', num2str(t)]);
xlabel('x'); ylabel('y');
print(h, 'mode.png', '-dpng', '-r150')

h1 = figure;
subplot(1,2,1)
loglog([0:N/2],mean(ke(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
title('kinetic energy spectrum'); xlabel('wavenumber');
subplot(1,2,2)
loglog([0:N/2],mean(ape(:,end-100:end),2),'.-', 'LineWidth',1); hold on;
title('potential energy spectrum'); xlabel('wavenumber');
print(h1, 'energy.png', '-dpng', '-r150')

% exit