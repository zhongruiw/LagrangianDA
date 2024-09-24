function [ENE, ENS] = Spectrum_layer(q_hat,p)
% Function takes Fourier coefficients of PV (q_hat) and struct containing
% parameters (p) and evaluates energy and enstrophy 1D spectra
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global hk
% persistent dX DX DY Laplacian InvBT InvBC KX KY
% if isempty(DX)
    k = [0:p.N/2 -p.N/2+1:-1]';
    [KX, KY] = meshgrid(k,k);
    dX = 1i*repmat(k',[p.N 1]);
    dY = 1i*repmat(k,[1 p.N]);
    Laplacian = dX(:,:).^2+dY(:,:).^2;
    InvBT = 1./Laplacian; InvBT(1,1) = 0;

    clear k
% end

% Invert for psi
q_bt = q_hat(:,:)+hk(:,:);
psi_hat = InvBT.*q_hat;


ENE = (KX.^2+KY.^2).*abs(psi_hat).^2;
ENS = abs(q_bt).^2;

ENE = .5*ENE/(p.N^4);
ENS = .5*ENS/(p.N^4);