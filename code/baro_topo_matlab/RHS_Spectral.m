function RHS = RHS_Spectral(q_hat,psi_hat,Ut,p)
% Function takes Fourier coefficients of PV (q_hat) and struct containing
% parameters (p) and evaluates RHS of barotropic QG equations except for
% high-k dissipation. Returns Fourier coefficients of RHS.
% Jacobian is dealiased using a 3/2 rule.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global hk
persistent dX DX dY DY %Laplacian InvBT
if isempty(DX)
    k = [0:p.N/2 -p.N/2+1:-1]';
    dX = 1i*repmat(k',[p.N 1]);
    dY = 1i*repmat(k,[1 p.N]);
    Laplacian = dX(:,:).^2+dY(:,:).^2;
    InvBT = 1./Laplacian; InvBT(1,1) = 0;
    
    k = [0:p.N/2-1 0 -p.N/2+1:-1]';
    dX = 1i*repmat(k',[p.N 1]);
    dY = 1i*repmat(k,[1 p.N]);
    % For the dealiased jacobian:
    k = [0:.75*p.N-1 0 -.75*p.N+1:-1]';
    DX = 1i*repmat(k',[1.5*p.N 1]);
    DY = 1i*repmat(k,[1 1.5*p.N]);
    clear k
end

% Invert for psi
q_bt = q_hat(:,:)+hk(:,:);
%psi_hat = InvBT.*q_hat;

% calculate Ekman plus beta plus mean shear
RHS = zeros([p.N p.N]);
RHS(:,:) = -Ut*dX(:,:).*q_bt(:,:)-(p.beta)*dX(:,:).*psi_hat(:,:);
    
% For using a 3/2-rule dealiased jacobian:
    % physical space, 3/2 grid; factor of (9/4) scales fft
    Psi_hat = zeros([1.5*p.N 1.5*p.N]);
    Psi_hat(1:p.N/2+1,1:p.N/2+1) = (9/4)*psi_hat(1:p.N/2+1,1:p.N/2+1);
    Psi_hat(1:p.N/2+1,p.N+2:1.5*p.N) = (9/4)*psi_hat(1:p.N/2+1,p.N/2+2:p.N);
    Psi_hat(p.N+2:1.5*p.N,1:p.N/2+1) = (9/4)*psi_hat(p.N/2+2:p.N,1:p.N/2+1);
    Psi_hat(p.N+2:1.5*p.N,p.N+2:1.5*p.N) = (9/4)*psi_hat(p.N/2+2:p.N,p.N/2+2:p.N);
    Q_hat = zeros([1.5*p.N 1.5*p.N]);
    Q_hat(1:p.N/2+1,1:p.N/2+1) = (9/4)*q_bt(1:p.N/2+1,1:p.N/2+1);
    Q_hat(1:p.N/2+1,p.N+2:1.5*p.N) = (9/4)*q_bt(1:p.N/2+1,p.N/2+2:p.N);
    Q_hat(p.N+2:1.5*p.N,1:p.N/2+1) = (9/4)*q_bt(p.N/2+2:p.N,1:p.N/2+1);
    Q_hat(p.N+2:1.5*p.N,p.N+2:1.5*p.N) = (9/4)*q_bt(p.N/2+2:p.N,p.N/2+2:p.N);
    % calculate u.gradq on 3/2 grid
    u = real(ifft2(-DY.*Psi_hat));
    v = real(ifft2( DX.*Psi_hat));
    qx= real(ifft2( DX.*Q_hat));
    qy= real(ifft2( DY.*Q_hat));
    jaco_real = u.*qx+v.*qy;
    % fft, 3/2 grid; factor of (4/9) scales fft
    Jaco_hat = (4/9)*fft2(jaco_real);
    % reduce to normal grid
    jaco_hat = zeros([p.N p.N]);
    jaco_hat(1:p.N/2+1,1:p.N/2+1) = Jaco_hat(1:p.N/2+1,1:p.N/2+1);
    jaco_hat(1:p.N/2+1,p.N/2+2:p.N) = Jaco_hat(1:p.N/2+1,p.N+2:1.5*p.N);
    jaco_hat(p.N/2+2:p.N,1:p.N/2+1) = Jaco_hat(p.N+2:1.5*p.N,1:p.N/2+1);
    jaco_hat(p.N/2+2:p.N,p.N/2+2:p.N) = Jaco_hat(p.N+2:1.5*p.N,p.N+2:1.5*p.N);
    
% Put it all together
RHS = RHS - jaco_hat;
