function [RHS, psi_hat] = RHS_Spectral_topo(q_hat,p,Ut)
% Function takes Fourier coefficients of PV (q_hat) and struct containing
% parameters (p) and evaluates RHS of 2-layer QG equations except for
% high-k dissipation. Returns Fourier coefficients of RHS.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global hk
persistent dX dY DX DY Laplacian InvBT InvBC
if isempty(DX)
    k = [0:p.N/2 -p.N/2+1:-1]';
    dX = 1i*repmat(k',[p.N 1 2]);
    dY = 1i*repmat(k,[1 p.N 2]);
    Laplacian = dX(:,:,1).^2+dY(:,:,1).^2;
    InvBT = 1./Laplacian; InvBT(1,1) = 0;
    InvBC = 1./(Laplacian-p.kd^2);InvBC(1,1) = 0;
    k = [0:p.N/2-1 0 -p.N/2+1:-1]';
    dX = 1i*repmat(k',[p.N 1 2]);
    dY = 1i*repmat(k,[1 p.N 2]);
    % For the dealiased jacobian:
    k = [0:.75*p.N-1 0 -.75*p.N+1:-1]';
    DX = 1i*repmat(k',[1.5*p.N 1 2]);
    DY = 1i*repmat(k,[1 1.5*p.N 2]);
    clear k
end

% Invert for psi
% Since we are using 'relative PV' for integration, we need to add topography to q_hat
% Alternatively, we can use full PV in integration so that the following line is unnecessary.
% q_hat(:,:,2) = q_hat(:,:,2) + hk(:,:); % add topograpy to get a 'full PV' 
q_bt = .5*(q_hat(:,:,1) + q_hat(:,:,2));
q_bc = .5*(q_hat(:,:,1) - q_hat(:,:,2));
psi_bt = InvBT.*(q_bt - 0.5*hk(:,:)); % topography comes in here 
psi_bc = InvBC.*(q_bc + 0.5*hk(:,:)); 
psi_hat(:,:,2) = psi_bt-psi_bc;
psi_hat(:,:,1) = psi_bt+psi_bc;

% calculate Ekman plus beta plus mean shear
RHS = zeros([p.N p.N 2]);
RHS(:,:,1) =-Ut*dX(:,:,1).*q_hat(:,:,1)-(p.kb^2 + Ut*p.kd^2)*dX(:,:,1).*psi_hat(:,:,1); 
RHS(:,:,2) = Ut*dX(:,:,1).*q_hat(:,:,2)-(p.kb^2 - Ut*p.kd^2)*dX(:,:,1).*psi_hat(:,:,2) - (p.r*Laplacian.*psi_hat(:,:,2) - Ut*dX(:,:,1).*hk(:,:)); % dX.*hk can be stored to accelerate the code
   
% rhs_1_1 = -Ut*dX(:,:,1).*q_hat(:,:,1);
% rhs_1_2 = -(p.kb^2 + Ut*p.kd^2)*dX(:,:,1).*psi_hat(:,:,1);
% rhs_2_3 = p.r*Laplacian.*psi_hat(:,:,2);
% rhs_2_4 = - Ut*dX(:,:,1).*hk(:,:);

% For using a 3/2-rule dealiased jacobian:
    % physical space, 3/2 grid; factor of (9/4) scales fft
    Psi_hat = zeros([1.5*p.N 1.5*p.N 2]);
    Psi_hat(1:p.N/2+1,1:p.N/2+1,:) = (9/4)*psi_hat(1:p.N/2+1,1:p.N/2+1,:);
    Psi_hat(1:p.N/2+1,p.N+2:1.5*p.N,:) = (9/4)*psi_hat(1:p.N/2+1,p.N/2+2:p.N,:);
    Psi_hat(p.N+2:1.5*p.N,1:p.N/2+1,:) = (9/4)*psi_hat(p.N/2+2:p.N,1:p.N/2+1,:);
    Psi_hat(p.N+2:1.5*p.N,p.N+2:1.5*p.N,:) = (9/4)*psi_hat(p.N/2+2:p.N,p.N/2+2:p.N,:);
    Q_hat = zeros([1.5*p.N 1.5*p.N 2]);
    Q_hat(1:p.N/2+1,1:p.N/2+1,:) = (9/4)*q_hat(1:p.N/2+1,1:p.N/2+1,:);
    Q_hat(1:p.N/2+1,p.N+2:1.5*p.N,:) = (9/4)*q_hat(1:p.N/2+1,p.N/2+2:p.N,:);
    Q_hat(p.N+2:1.5*p.N,1:p.N/2+1,:) = (9/4)*q_hat(p.N/2+2:p.N,1:p.N/2+1,:);
    Q_hat(p.N+2:1.5*p.N,p.N+2:1.5*p.N,:) = (9/4)*q_hat(p.N/2+2:p.N,p.N/2+2:p.N,:);
    % calculate u.gradq on 3/2 grid
    u = real(ifft2(-DY.*Psi_hat));
    v = real(ifft2( DX.*Psi_hat));
    qx= real(ifft2( DX.*Q_hat));
    qy= real(ifft2( DY.*Q_hat));
    jaco_real = u.*qx+v.*qy;
    % fft, 3/2 grid; factor of (4/9) scales fft
    Jaco_hat = (4/9)*fft2(jaco_real);
    % reduce to normal grid
    jaco_hat = zeros([p.N p.N 2]);
    jaco_hat(1:p.N/2+1,1:p.N/2+1,:) = Jaco_hat(1:p.N/2+1,1:p.N/2+1,:);
    jaco_hat(1:p.N/2+1,p.N/2+2:p.N,:) = Jaco_hat(1:p.N/2+1,p.N+2:1.5*p.N,:);
    jaco_hat(p.N/2+2:p.N,1:p.N/2+1,:) = Jaco_hat(p.N+2:1.5*p.N,1:p.N/2+1,:);
    jaco_hat(p.N/2+2:p.N,p.N/2+2:p.N,:) = Jaco_hat(p.N+2:1.5*p.N,p.N+2:1.5*p.N,:);

% Put it all together
RHS = RHS - jaco_hat;
