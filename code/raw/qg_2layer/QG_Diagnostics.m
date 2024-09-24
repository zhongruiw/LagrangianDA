function [vb,utz] = QG_Diagnostics(q_hat,p)
% Function takes Fourier coefficients of q and computes
% area-integrated meridional heat flux
%   vb := .5*kd^2*int((psi_2)_x psi_1)
% (Note that this equals barotropic v times baroclinic streamfunction up to
% the factor of kd^2.)
% and it returns zonally-averaged barotropic (utz) zonal velocity.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

% Invert for psi
psi_bt = .5*InvBT.*(q_hat(:,:,1) + q_hat(:,:,2));
psi_bc = .5*InvBC.*(q_hat(:,:,1) - q_hat(:,:,2));

% Real-Space quantities
vt = real(ifft2(dX.*psi_bt));
psic = real(ifft2(psi_bc));
ut = real(ifft2(-dY.*psi_bt));

% Outputs
vb  = ((2*pi*p.kd/p.N)^2)*sum(sum(vt.*psic));
utz = mean(ut,2);