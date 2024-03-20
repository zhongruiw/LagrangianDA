function [vb,utz,e, etp] = QG_Diagnostics_topo(q_hat,p)
% Function takes Fourier coefficients of q and computes
% area-integrated meridional heat flux
%   vb := .5*kd^2*int((psi_2)_x psi_1)
% (Note that this equals barotropic v times baroclinic streamfunction up to
% the factor of kd^2.)
% and it returns zonally-averaged barotropic (utz) zonal velocity.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global hk
k = [0:p.N/2 -p.N/2+1:-1]';
dX = 1i*repmat(k',[p.N 1]);
dY = 1i*repmat(k,[1 p.N]);
Laplacian = dX.^2+dY.^2;
InvBT = 1./Laplacian; InvBT(1,1) = 0;
InvBC = 1./(Laplacian-p.kd^2);InvBC(1,1) = 0;
k = [0:p.N/2-1 0 -p.N/2+1:-1]';
dX = 1i*repmat(k',[p.N 1]);
dY = 1i*repmat(k,[1 p.N]);
clear k
% 
% mu1 = -p.kb^2/p.U - p.kd^2;
% mu2 = p.kb^2/p.U - p.kd^2;

% Invert for psi
% q_hat(:,:,2) = q_hat(:,:,2) + hk(:,:); %%%%%%%%%%%%%%%%%%%%% 
q_bt = .5*(q_hat(:,:,1) + q_hat(:,:,2));
q_bc = .5*(q_hat(:,:,1) - q_hat(:,:,2));
psi_bt = InvBT.*(q_bt - 0.5*hk(:,:)); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psi_bc = InvBC.*(q_bc + 0.5*hk(:,:)); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Real-Space quantities
vt = real(ifft2(dX.*psi_bt));
vc = real(ifft2(dX.*psi_bc));
psic = real(ifft2(psi_bc));
ut = real(ifft2(-dY.*psi_bt));
uc = real(ifft2(-dY.*psi_bc));
q1 = real(ifft2(q_hat(:,:,1)));
q2 = real(ifft2(q_hat(:,:,2)));

% Outputs
vb  = ((2*pi*p.kd/p.N)^2)*sum(sum(vt.*psic));
utz = mean(ut,2);
e = (2*pi/p.N)^2 * sum(sum(ut.^2+vt.^2+uc.^2+ut.^2+p.kd^2*psic.^2));
etp = 1/2* (2*pi/p.N)^2 * sum(sum((q1.^2 + q2.^2)));