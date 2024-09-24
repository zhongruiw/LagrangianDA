% This script solves barotropic QG flow with topography in a doubly-periodic domain for the 57-mode model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set simulation parameters
Lamda = 64^2; %128^2;
N = (floor(sqrt(Lamda))+2)*2; % Number of points in each direction
dt = 2E-3; % initial time step size
Nt = 1E4; % Number of time steps
qlim = 1.5E5; % if any q > qlim, simulation stops


% Set physical parameters
gamma = 0.1;
beta = sqrt(gamma)*5;   % beta effect
r = 8;                  % hyperdiffusion exponent (need to be even)
nu = 0;                 % Coefficient of biharmonic vorticity diffusion
U = sqrt(gamma)*3;      % background mean flow
H = gamma;


% Set up hyperviscous PV dissipation
k = [0:N/2 -N/2+1:-1]'; % wavenumbers
dX = 1i*repmat(k',[N 1]);
dY = 1i*repmat(k,[1 N]);
Laplacian = dX(:,:).^2+dY(:,:).^2;
% mode elimination matrix
Lmod=zeros(N,N);
for jj=1:N
    for ii=1:N
        if k(ii)^2+k(jj)^2 <= Lamda
            Lmod(ii,jj)=1;
        end
    end
end
L = zeros([N N]);
for jj=1:N
    for ii=1:N
        kr = sqrt(k(ii)^2+k(jj)^2);
        L(ii,jj) = -nu*kr^r;
    end
end
L=L.*Lmod;
clear k kr ii jj

% Put useful stuff into a struct
params = struct('U',U, 'beta',beta, 'r',r, 'nu',nu, 'N',N, 'Lmod',Lmod, 'dt',dt, 'gamma',gamma);


% Initialize
t = 0;
% Initialize topography
dx=2*pi/N;
[X,Y]=meshgrid(-pi:dx:pi-dx,-pi:dx:pi-dx);
topo = H * (cos(X)+2*cos(2*X));
topo = topo-mean(mean(topo));
global hk 
hk = fft2(topo).*Lmod;
% initialize streamfunction
psi0=(gamma-2)*cos(X)+(gamma-1)/2*cos(2*X) + .005*(sin(Y+.24)+sin(2*X+Y+.97));
psik0=fft2(psi0).*Lmod;
q = Laplacian.*psik0;
% qp(:,:) = ifft2(q);
clear dX dY Laplacian

% qp(:,:) = 1*randn(params.N);
% q = fft2(qp);

% qp(:,:) = qp(:,:)-mean(mean(qp(:,:)));
Ut = params.U;


% Diagnostics
countDiag = 10; % Compute diagnostics every countDiag steps
T = zeros(1,Nt/countDiag);
Um = zeros(1,Nt/countDiag);
qk = zeros(N,N,Nt/countDiag);
pk = zeros(N,N,Nt/countDiag);
energy = zeros(N,N,Nt/countDiag);
enstrophy = zeros(N,N,Nt/countDiag);


% Main loop 
% tic;
for ii=1:Nt
    if mod(ii,countDiag)==0
        if any(isnan(q(:))), break, end
        T(ii/countDiag)=t;
        qk(:,:,ii/countDiag)=q;
        pk(:,:,ii/countDiag)=inv_laplacian(q,params);
        [ENE,ENS] = Spectrum_layer(q,params);
        energy(:,:,ii/countDiag) = ENE; enstrophy(:,:,ii/countDiag) = ENS;
        Um(:,ii/countDiag) = Ut;
%         toc
        if mod(ii,100)==0
           display(['iteration i = ', num2str(ii), '; energy E = ',num2str(sum(sum(ENE)))]);
        end

    end

    M = 1./(1-.25*dt*L);
    % First stage ARK4
    psik = inv_laplacian(q,params).*Lmod;
    k0 = RHS_Spectral(q,psik,Ut,params).*Lmod;
    l0 = L.*q;
    u0 = RHS_meanFlow(psik,params);
    % Second stage
    q1 = M.*(q+.5*dt*k0+.25*dt*l0);
    U1 = Ut+.5*dt*u0;
    psik = inv_laplacian(q1,params).*Lmod;
    k1 = RHS_Spectral(q1,psik,U1,params).*Lmod;
    l1 = L.*q1;
    u1 = RHS_meanFlow(psik,params);
    % Third stage
    q2 = M.*(q+dt*(13861*k0/62500+6889*k1/62500+8611*l0/62500-1743*l1/31250));
    U2 = Ut +  dt*(13861*u0/62500+6889*u1/62500);
    psik = inv_laplacian(q2,params).*Lmod;
    k2 = RHS_Spectral(q2,psik,U2,params).*Lmod;
    l2 = L.*q2;
    u2 = RHS_meanFlow(psik,params);
    % Fourth stage
    q3 = M.*(q+dt*(-0.04884659515311858*k0-0.1777206523264010*k1+0.8465672474795196*k2...
    +0.1446368660269822*l0-0.2239319076133447*l1+0.4492950415863626*l2));
    U3 = Ut +  dt*(-0.04884659515311858*u0-0.1777206523264010*u1+0.8465672474795196*u2);
    psik = inv_laplacian(q3,params).*Lmod;
    k3 = RHS_Spectral(q3,psik,U3,params).*Lmod;
    l3 = L.*q3;
    u3 = RHS_meanFlow(psik,params);
    % Fifth stage
    q4 = M.*(q+dt*(-0.1554168584249155*k0-0.3567050098221991*k1+1.058725879868443*k2...
    +0.3033959883786719*k3+0.09825878328356477*l0-0.5915442428196704*l1...
    +0.8101210538282996*l2+0.2831644057078060*l3));
    U4 = Ut +  dt*(-0.1554168584249155*u0-0.3567050098221991*u1+1.058725879868443*u2...
    +0.3033959883786719*u3);
    psik = inv_laplacian(q4,params).*Lmod;
    k4 = RHS_Spectral(q4,psik,U4,params).*Lmod;
    l4 = L.*q4;
    u4 = RHS_meanFlow(psik,params);
    % Sixth stage
    q5 = M.*(q+dt*(0.2014243506726763*k0+0.008742057842904184*k1+0.1599399570716811*k2...
    +0.4038290605220775*k3+0.2260645738906608*k4+0.1579162951616714*l0...
    +0.1867589405240008*l2+0.6805652953093346*l3-0.2752405309950067*l4));
    U5 = Ut +  dt*(0.2014243506726763*u0+0.008742057842904184*u1+0.1599399570716811*u2...
    +0.4038290605220775*u3+0.2260645738906608*u4);
    psik = inv_laplacian(q5,params).*Lmod;
    k5 = RHS_Spectral(q5,psik,U5,params).*Lmod;
    l5 = L.*q5;
    u5 = RHS_meanFlow(psik,params);


    % Successful step, proceed to evaluation
    t = t+dt;
    qp = real(ifft2(q+dt*(0.1579162951616714*(k0+l0)+0.1867589405240008*(k2+l2)+...
    0.6805652953093346*(k3+l3)-0.2752405309950067*(k4+l4)+(k5+l5)/4)));
    q = fft2(qp).*Lmod;    
    Ut = Ut+dt*(0.1579162951616714*(u0)+0.1867589405240008*(u2)+...
    0.6805652953093346*(u3)-0.2752405309950067*(u4)+(u5)/4);

    if any(abs(qp(:))>qlim),break,end
end

[xx,yy] = meshgrid(linspace(-pi,pi,params.N));
figure
contour(xx,yy,qp(:,:),50); colorbar;
title(['barotropic mode at t = ', num2str(t)]);
xlabel('x'); ylabel('y');

save(['./57model1_all_U',num2str(U/sqrt(gamma)),'H0',num2str(H*100)]);
