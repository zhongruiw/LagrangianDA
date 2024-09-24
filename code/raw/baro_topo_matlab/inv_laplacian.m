function psi_hat = inv_laplacian(q_hat,p)
% Function takes Fourier coefficients of PV (q_hat) back to 
% the Fourier coefficients of stream function (psi_hat).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

persistent dX dY Laplacian InvBT
if isempty(dX)
    k = [0:p.N/2 -p.N/2+1:-1]';
    dX = 1i*repmat(k',[p.N 1]);
    dY = 1i*repmat(k,[1 p.N]);
    Laplacian = dX(:,:).^2+dY(:,:).^2;
    InvBT = 1./Laplacian; InvBT(1,1) = 0;

    clear k
end

% Invert for psi
psi_hat = InvBT.*q_hat;