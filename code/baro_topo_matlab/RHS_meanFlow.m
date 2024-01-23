function RHS = RHS_meanFlow(psi_hat,p)
% Function takes Fourier coefficients of PV (q_hat) and struct containing
% parameters (p) and evaluates RHS of the mean dynamical flow equation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global hk
persistent dX
if isempty(dX)
    k = [0:p.N/2 -p.N/2+1:-1]';
    dX = 1i*repmat(k',[p.N 1]);
    clear k
end


% calculate topographic stree
RHS = sum(sum(dX(:,:).*psi_hat(:,:).*conj(hk(:,:))));
RHS = real(RHS)/(p.N^4);