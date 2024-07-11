% Comparing two DA method using the CGNS as a test model
% Smoother is used for both tests
% The test model is:
% dx = (-x + 2y) dt + sigma_x d Wx
% dy = (-2y + z) dt + sigma_y d Wy
% dz = (y - 2z) dt + sigma_z d Wz
% where x is the observed variable
% The goal is to compare the DA result in recovering z
% The linear system is chosen so everything can be explicitly computed
% Method I: Direct DA, computing p(y(t),z(t)|x([0,T]))
% Method II: Two-step DA, (a) computing p(y(t)|x([0,t])). Here we can still
% start with computing p(y(t),z(t)|x([0,T])) and taking the marignal
% distribution. (b) sampling % multiple trajectory of y, 
% computing p(z(t)|y[0,T]) for each trajectory
% and then computing the Gaussian mixture distribution

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Generating the true signal %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% coefficient matrix in the linear equation
C1 = [-1,4,0;
    0,-2,1;
    0,1,-2];
C2 = [-1,4,0;
    0,-2,4;
    0,-4,-2];
C3 = [-1,4,0;
    0,-2,-1;
    0,1,-2];
C = C3;
% noise coefficients
sigma_x = .5;
sigma_y = 1;
sigma_z = 1;
% Generate the true signal
rng(1);
N = 20000; % total nubmer of time steps
dt = 0.005; % time step
x = zeros(1,N);
y = zeros(1,N);
z = zeros(1,N);
lag = 3000; % lag for computing the ACF
% numerical integration: Eulerian-Maruyama method
for i = 2:N
    x(i) = x(i-1) + ( C(1,1) * x(i-1) + C(1,2) * y(i-1) + C(1,3) * z(i-1) ) * dt + sigma_x * sqrt(dt) * randn;
    y(i) = y(i-1) + ( C(2,1) * x(i-1) + C(2,2) * y(i-1) + C(2,3) * z(i-1) ) * dt + sigma_y * sqrt(dt) * randn;
    z(i) = z(i-1) + ( C(3,1) * x(i-1) + C(3,2) * y(i-1) + C(3,3) * z(i-1) ) * dt + sigma_z * sqrt(dt) * randn;
end
disp('Correlation between y and z')
corcf = corrcoef(y,z);
disp(corcf(1,2))
% Showing the true signal and the ACF for each variable
figure
for i = 1:3
    subplot(3,4,[1:3]+(i-1)*4)
    if i == 1
        variable = x;
    elseif i == 2
        variable = y;
    else
        variable = z;
    end    
    plot(dt:dt:N*dt, variable,'b','linewidth',2);
    box on
    set(gca,'fontsize',12)
    if i == 1
        variable = x;
        title('True signal of x')
    elseif i == 2
        variable = y;
        title('True signal of y')
    else
        variable = z;
        title('True signal of z')
        xlabel('t')
    end    
    subplot(3,4,4+(i-1)*4)
    ACF = autocorr(variable, lag);
    plot(0:dt:lag*dt,ACF,'b','linewidth',2)
    box on
    set(gca,'fontsize',12)
    if i == 1
        variable = x;
        title('ACF of x')
    elseif i == 2
        variable = y;
        title('ACF of y')
    else
        variable = z;
        title('ACF of z')
        xlabel('t')
    end    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Method I %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

InvBoB = 1/sigma_x^2;
b1 = [sigma_y,0;0,sigma_z];
mu0 = [0;0]; % initial value of posterior mean
n = 2;
R0 = eye(n,n); % initial value of posterior covariance
u_post_mean = zeros(n,N); % posterior mean
u_post_mean(:,1) = mu0;
u_post_cov = zeros(n,n,N); % posterior covariance
u_post_cov(:,:,1) = R0; % only save the diagonal elements
for i = 2:N
    x0 = x(i-1); 
    x1 = x(i); 
    a0 = [C(2,1) * x0;
        C(3,1) * x0];
    a1 = [C(2,2), C(2,3);
        C(3,2), C(3,3)];
    % matrix for filtering
    A0 = C(1,1) * x0;
    A1 = [C(1,2), C(1,3)];
    % update the posterior mean and posterior covariance
    mu = mu0 + (a0 + a1 * mu0) * dt + (R0 * A1') * InvBoB * (x1-x0 - A1 * mu0 * dt);
    R = R0 + (a1 * R0 + R0* a1' + b1*b1' - (R0*A1') * InvBoB * (R0*A1')')*dt;
    u_post_mean(:,i) = mu;
    u_post_cov(:,:,i) = R;
    mu0 = mu;
    R0 = R;  
end
gamma_mean_trace = u_post_mean;
gamma_cov_trace = u_post_cov;

mu_s = zeros(n,N); % Save the posterior mean in smoothing
R_s = zeros(n,n,N); % Save the posterior covariance in smoothing

% Smoothing is backward
% Intial values for smoothing (at the last time instant)
mu_s(:,end) = mu; % save the initial value of the smoother mean 
R_s(:,:,end) = R; % save the initial value of the smoother covariance
Y_Sampling = zeros(n,N); % Save for the backward sampled trajectory
% rd_Y = randn(n,N); % pre-generated random numbers
s_n = 20;
Y_Sampling_Save = zeros(n,N,s_n);

for i = N-1:-1:1
    % Matrices and vectors in the conditional Gaussian smoothing and
    % backward sampling
 
    gamma_cov = gamma_cov_trace(:,:,i); % filter covariance is needed as the input of smoothing formula 
    C_temp = gamma_cov * (eye(n) + a1 * dt)' * (b1 * b1' * dt + (eye(n) + a1 * dt) * gamma_cov * (eye(n) + a1 * dt)')^(-1);
    mu_s(:,i) = gamma_mean_trace(:,i) + C_temp * (mu_s(:,i+1) - a0 * dt - ( eye(n) + a1 * dt) * gamma_mean_trace(:,i)); % update the smoother mean
    R_s_temp = R_s(:,:,i+1);
    R_s_temp = gamma_cov + C_temp * (R_s_temp - (eye(n) + a1 * dt) * gamma_cov * (eye(n) + a1 * dt)' - b1 * b1' * dt) * C_temp';   
    R_s(:,:,i) = R_s_temp; % update the smoother covariance (this line and the above two lines)
    for j = 1:s_n
        Y_Sampling_Save(:,i,j) = Y_Sampling_Save(:,i+1,j) + (-a0 - a1 * Y_Sampling_Save(:,i+1,j)) * dt + b1 * b1' * inv(gamma_cov) * (gamma_mean_trace(:,i) ... 
        - Y_Sampling_Save(:,i+1,j)) * dt + b1 * randn(n,1) * sqrt(dt); % Backward sampling; the sampled trajectory has random noise
    end

end
y_sampled = squeeze(Y_Sampling_Save(1,:,:));

u_post_mean = mu_s;
u_post_cov = R_s;
u_post_mean_one_step = mu_s;
u_post_cov_one_step = R_s;
figure
for i = 1:2
    subplot(4,1,i)
    if i == 1
        variable = y;
    elseif i == 2
        variable = z;
    end    
    hold on
    plot(dt:dt:N*dt, variable,'b','linewidth',2);
    plot(dt:dt:N*dt, gamma_mean_trace(i,:),'r','linewidth',2);
    post_upper = gamma_mean_trace(i,:) + 2 * transpose(sqrt(squeeze(gamma_cov_trace(i,i,:))));
    post_lower = gamma_mean_trace(i,:) - 2 * transpose(sqrt(squeeze(gamma_cov_trace(i,i,:))));
    tt = dt:dt:N*dt;
    patch([tt,tt(end:-1:1)],[post_lower,post_upper(end:-1:1)],'r','facealpha',0.2,'linestyle','none');
    box on
    set(gca,'fontsize',12)
end
for i = 1:2
    subplot(4,1,i+2)
    if i == 1
        variable = y;
    elseif i == 2
        variable = z;
    end    
    hold on
    plot(dt:dt:N*dt, variable,'b','linewidth',2);
    plot(dt:dt:N*dt, u_post_mean(i,:),'r','linewidth',2);
    post_upper = u_post_mean(i,:) + 2 * transpose(sqrt(squeeze(u_post_cov(i,i,:))));
    post_lower = u_post_mean(i,:) - 2 * transpose(sqrt(squeeze(u_post_cov(i,i,:))));
    tt = dt:dt:N*dt;
    patch([tt,tt(end:-1:1)],[post_lower,post_upper(end:-1:1)],'r','facealpha',0.2,'linestyle','none');
    box on
    set(gca,'fontsize',12)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Method II %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Here the sampled trajectories are from the data assimilation results in
% Method I. So, there is no model error in getting these time series. In
% practice, a different model would be used in the DA of part of the system
% and therefore model error may exist. 

u_post_mean_all = zeros(s_n,N);
u_post_cov_all = zeros(s_n,N);
for k = 1:s_n
    InvBoB = 1/sigma_y^2;
    b1 = sigma_z;
    mu0 = 0; % initial value of posterior mean
    R0 = 1; % initial value of posterior covariance
    u_post_mean = zeros(1,N); % posterior mean
    u_post_mean(1) = mu0;
    u_post_cov = zeros(1,N); % posterior covariance
    u_post_cov(1) = R0; % only save the diagonal elements
    for i = 2:N
        y0 = y_sampled(i-1,k); 
        y1 = y_sampled(i,k); 
        a0 = C(3,2) * y0;
        a1 = C(3,3);
        % matrix for filtering
        A0 = C(2,2) * y0;
        A1 = C(2,3);
        % update the posterior mean and posterior covariance
        mu = mu0 + (a0 + a1 * mu0) * dt + (R0 * A1') * InvBoB * (y1-y0 - A1 * mu0 * dt);
        R = R0 + (a1 * R0 + R0* a1' + b1*b1' - (R0*A1') * InvBoB * (R0*A1')')*dt;
        u_post_mean(i) = mu;
        u_post_cov(i) = R;
        mu0 = mu;
        R0 = R;  
    end
    gamma_mean_trace = u_post_mean;
    gamma_cov_trace = u_post_cov;
    
    mu_s = zeros(1,N); % Save the posterior mean in smoothing
    R_s = zeros(1,N); % Save the posterior covariance in smoothing

    % Smoothing is backward
    % Intial values for smoothing (at the last time instant)
    mu_s(end) = mu; % save the initial value of the smoother mean 
    R_s(end) = R; % save the initial value of the smoother covariance
    n = 1;
    for i = N-1:-1:1
        % Matrices and vectors in the conditional Gaussian smoothing and
        % backward sampling

        gamma_cov = gamma_cov_trace(i); % filter covariance is needed as the input of smoothing formula 
        C_temp = gamma_cov * (eye(n) + a1 * dt)' * (b1 * b1' * dt + (eye(n) + a1 * dt) * gamma_cov * (eye(n) + a1 * dt)')^(-1);
        mu_s(i) = gamma_mean_trace(i) + C_temp * (mu_s(i+1) - a0 * dt - ( eye(n) + a1 * dt) * gamma_mean_trace(i)); % update the smoother mean
        R_s_temp = R_s(i+1);
        R_s_temp = gamma_cov + C_temp * (R_s_temp - (eye(n) + a1 * dt) * gamma_cov * (eye(n) + a1 * dt)' - b1 * b1' * dt) * C_temp';   
        R_s(i) = R_s_temp; % update the smoother covariance (this line and the above two lines)  
    end
    u_post_mean_all(k,:) = mu_s;
    u_post_cov_all(k,:) = R_s;
end
z_mean = mean(u_post_mean_all);
z_var = mean(u_post_cov_all) + var(u_post_mean_all);
figure
variable = z;
hold on
h1 = plot(dt:dt:N*dt, variable,'b','linewidth',2);
tt = dt:dt:N*dt;
h2 = plot(dt:dt:N*dt, u_post_mean_one_step(2,:),'r','linewidth',2);
post_upper_one_step = u_post_mean_one_step(2,:) + 2 * transpose(sqrt(squeeze(u_post_cov_one_step(2,2,:))));
post_lower_one_step = u_post_mean_one_step(2,:) - 2 * transpose(sqrt(squeeze(u_post_cov_one_step(2,2,:))));
patch([tt,tt(end:-1:1)],[post_lower_one_step,post_upper_one_step(end:-1:1)],'r','facealpha',0.2,'linestyle','none');
h3 = plot(dt:dt:N*dt, z_mean,'g','linewidth',2);
post_upper = z_mean + 2 * sqrt(z_var);
post_lower = z_mean - 2 * sqrt(z_var);
patch([tt,tt(end:-1:1)],[post_lower,post_upper(end:-1:1)],'g','facealpha',0.2,'linestyle','none');
box on
set(gca,'fontsize',12)
legend([h1,h2,h3],'Truth','Method I','Method II')


RMSE_I = sqrt(sum((z - u_post_mean_one_step(2,:)).^2)/N);
RMSE_II = sqrt(sum((z - z_mean).^2)/N);
disp('RMSEs of Method I and Method II')
disp([RMSE_I, RMSE_II])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Method III %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Here the sampled trajectories are from the data assimilation results in
% Method I. So, there is no model error in getting these time series. In
% practice, a different model would be used in the DA of part of the system
% and therefore model error may exist. 
% A different method p(z|x) = p(y|x)p(z|y,x)
% Here p(z|y,x) is different from Method II, which was p(z|y)

u_post_mean_all = zeros(s_n,N);
u_post_cov_all = zeros(s_n,N);
for k = 1:s_n
    InvBoB = [1/sigma_x^2,0;0,1/sigma_y^2];
    b1 = sigma_z;
    mu0 = 0; % initial value of posterior mean
    R0 = 1; % initial value of posterior covariance
    u_post_mean = zeros(1,N); % posterior mean
    u_post_mean(1) = mu0;
    u_post_cov = zeros(1,N); % posterior covariance
    u_post_cov(1) = R0; % only save the diagonal elements
    for i = 2:N
        x0 = x(i-1); 
        x1 = x(i); 
        y0 = y_sampled(i-1,k); 
        y1 = y_sampled(i,k); 
        obs_diff = [x1 - x0; y1 - y0];
        a0 = C(3,1) * x0 + C(3,2) * y0;
        a1 = C(3,3);
        % matrix for filtering
        A0 = [C(1,1) * x0 + C(1,2) * y0;
            C(2,1) * x0 + C(2,2) * y0];
        A1 = [C(1,3);C(2,3)];
        % update the posterior mean and posterior covariance
        mu = mu0 + (a0 + a1 * mu0) * dt + (R0 * A1') * InvBoB * (obs_diff - A1 * mu0 * dt);
        R = R0 + (a1 * R0 + R0* a1' + b1*b1' - (R0*A1') * InvBoB * (R0*A1')')*dt;
        u_post_mean(i) = mu;
        u_post_cov(i) = R;
        mu0 = mu;
        R0 = R;  
    end
    gamma_mean_trace = u_post_mean;
    gamma_cov_trace = u_post_cov;
    
    mu_s = zeros(1,N); % Save the posterior mean in smoothing
    R_s = zeros(1,N); % Save the posterior covariance in smoothing

    % Smoothing is backward
    % Intial values for smoothing (at the last time instant)
    mu_s(end) = mu; % save the initial value of the smoother mean 
    R_s(end) = R; % save the initial value of the smoother covariance
    n = 1;
    for i = N-1:-1:1
        % Matrices and vectors in the conditional Gaussian smoothing and
        % backward sampling

        gamma_cov = gamma_cov_trace(i); % filter covariance is needed as the input of smoothing formula 
        C_temp = gamma_cov * (eye(n) + a1 * dt)' * (b1 * b1' * dt + (eye(n) + a1 * dt) * gamma_cov * (eye(n) + a1 * dt)')^(-1);
        mu_s(i) = gamma_mean_trace(i) + C_temp * (mu_s(i+1) - a0 * dt - ( eye(n) + a1 * dt) * gamma_mean_trace(i)); % update the smoother mean
        R_s_temp = R_s(i+1);
        R_s_temp = gamma_cov + C_temp * (R_s_temp - (eye(n) + a1 * dt) * gamma_cov * (eye(n) + a1 * dt)' - b1 * b1' * dt) * C_temp';   
        R_s(i) = R_s_temp; % update the smoother covariance (this line and the above two lines)  
    end
    u_post_mean_all(k,:) = mu_s;
    u_post_cov_all(k,:) = R_s;
end
z_mean2 = mean(u_post_mean_all);
z_var2 = mean(u_post_cov_all) + var(u_post_mean_all);
figure
variable = z;
hold on
h1 = plot(dt:dt:N*dt, variable,'b','linewidth',2);
tt = dt:dt:N*dt;
h2 = plot(dt:dt:N*dt, u_post_mean_one_step(2,:),'r','linewidth',2);
post_upper_one_step = u_post_mean_one_step(2,:) + 2 * transpose(sqrt(squeeze(u_post_cov_one_step(2,2,:))));
post_lower_one_step = u_post_mean_one_step(2,:) - 2 * transpose(sqrt(squeeze(u_post_cov_one_step(2,2,:))));
patch([tt,tt(end:-1:1)],[post_lower_one_step,post_upper_one_step(end:-1:1)],'r','facealpha',0.2,'linestyle','none');
h3 = plot(dt:dt:N*dt, z_mean2,'k','linewidth',2);
post_upper = z_mean2 + 2 * sqrt(z_var2);
post_lower = z_mean2 - 2 * sqrt(z_var2);
patch([tt,tt(end:-1:1)],[post_lower,post_upper(end:-1:1)],'k','facealpha',0.2,'linestyle','none');
box on
set(gca,'fontsize',12)
legend([h1,h2,h3],'Truth','Method I','Method III')


RMSE_I = sqrt(sum((z - u_post_mean_one_step(2,:)).^2)/N);
RMSE_II = sqrt(sum((z - z_mean2).^2)/N);
disp('RMSEs of Method I and Method III')
disp([RMSE_I, RMSE_II])