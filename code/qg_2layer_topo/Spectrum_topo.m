function [KE, APE, E, ETP] = Spectrum_topo(q_hat,p)
% Function takes Fourier coefficients of PV (q_hat) and struct containing
% parameters (p) and evaluates KE and APE 1D spectra
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global hk
persistent dX DX DY Laplacian InvBT InvBC KX KY
if isempty(DX)
    k = [0:p.N/2 -p.N/2+1:-1]';
    [KX, KY] = meshgrid(k,k);
    dX = 1i*repmat(k',[p.N 1 2]);
    dY = 1i*repmat(k,[1 p.N 2]);
    Laplacian = dX(:,:,1).^2+dY(:,:,1).^2;
    InvBT = 1./Laplacian; InvBT(1,1) = 0;
    InvBC = 1./(Laplacian-p.kd^2);InvBC(1,1) = 0;
    k = [0:p.N/2-1 0 -p.N/2+1:-1]';
    dX = 1i*repmat(k',[p.N 1 2]);    
    dY = 1i*repmat(k,[1 p.N 2]);
    % k = [0:.75*p.N-1 0 -.75*p.N+1:-1]';
    % DX = 1i*repmat(k',[1.5*p.N 1 2]);
    % DY = 1i*repmat(k,[1 1.5*p.N 2]);
    clear k
end

% mu1 = -p.kb^2/p.U - p.kd^2;
% mu2 = p.kb^2/p.U - p.kd^2;

% Invert for psi
% q_hat(:,:,2) = q_hat(:,:,2) + hk(:,:); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
q_bt = .5*(q_hat(:,:,1) + q_hat(:,:,2));
q_bc = .5*(q_hat(:,:,1) - q_hat(:,:,2));
psi_bt = InvBT.*(q_bt - 0.5*hk(:,:)); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psi_bc = InvBC.*(q_bc + 0.5*hk(:,:)); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
psi_hat(:,:,2) = psi_bt-psi_bc;
psi_hat(:,:,1) = psi_bt+psi_bc;
DX_psi_hat = dX.*psi_hat;
DY_psi_hat = dY.*psi_hat;

% 
KE = zeros(p.N/2+1,1);
APE = KE;
E = zeros(p.N/2+1,1); % energy
ETP = zeros(p.N/2+1,1); % enstrophy
for jj=1:p.N
    for ii=1:p.N
        k = sqrt(KX(ii,jj).^2+KY(ii,jj).^2); 
        if ceil(k)<=p.N/2
            r = k-floor(k);
            KE(floor(k)+1) = KE(floor(k)+1)+(1-r)*(k^2)*(abs(psi_hat(ii,jj,1))^2+abs(psi_hat(ii,jj,2))^2);
            APE(floor(k)+1)= APE(floor(k)+1)+(1-r)*(.5*p.kd^2)*abs(psi_hat(ii,jj,1)-psi_hat(ii,jj,2))^2;
            E(floor(k)+1)= E(floor(k)+1)+(1-r)*(abs(DX_psi_hat(ii,jj,1))^2+abs(DY_psi_hat(ii,jj,1))^2+abs(DX_psi_hat(ii,jj,2))^2+abs(DY_psi_hat(ii,jj,2))^2+(.5*p.kd^2)*(psi_hat(ii,jj,1)-psi_hat(ii,jj,2))^2);
            ETP(floor(k)+1)= ETP(floor(k)+1)+(1-r)*(abs(q_hat(ii,jj,1))^2 + abs(q_hat(ii,jj,2))^2);
            KE(ceil(k)+1) = KE(ceil(k)+1)+r*(k^2)*(abs(psi_hat(ii,jj,1))^2+abs(psi_hat(ii,jj,2))^2);
            APE(ceil(k)+1)= APE(ceil(k)+1)+r*(.5*p.kd^2)*abs(psi_hat(ii,jj,1)-psi_hat(ii,jj,2))^2;
            E(ceil(k)+1)= E(ceil(k)+1)+r*(abs(DX_psi_hat(ii,jj,1))^2+abs(DY_psi_hat(ii,jj,1))^2+abs(DX_psi_hat(ii,jj,2))^2+abs(DY_psi_hat(ii,jj,2))^2+(.5*p.kd^2)*(psi_hat(ii,jj,1)-psi_hat(ii,jj,2))^2);
            ETP(ceil(k)+1)= ETP(ceil(k)+1)+r*(abs(q_hat(ii,jj,1))^2 + abs(q_hat(ii,jj,2))^2);
        end
    end
end
KE = .5*KE/(p.N^4);
APE = .5*APE/(p.N^4);
E = .5*E/(p.N^4);
ETP = .5*ETP/(p.N^4);
