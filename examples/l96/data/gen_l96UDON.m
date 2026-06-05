% Parameters
N = 40;
F = 6;

% Narrower time domain for DeepONet window training
tspan = [0, 1.0]; 
num_points = 101; 
t = linspace(tspan(1), tspan(2), num_points);

% Number of Initial Conditions to sample
num_ics = 500; 

% Preallocate arrays to store the dataset
usol_all = zeros(num_ics, num_points, N);
u0_all = zeros(num_ics, N);

opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-14);

disp('Generating trajectories...');
for i = 1:num_ics
    % Randomly sample initial conditions within a typical L96 range
    % L96 values typically oscillate roughly between -10 and +15
    u0 = -10 + 25 * rand(N, 1);
    
    u0_all(i, :) = u0';
    
    % Solve directly on the target time points
    [~, u] = ode113(@(t, u) lorenz96(t, u, N, F), t, u0, opts);
    
    usol_all(i, :, :) = u;
    
    if mod(i, 50) == 0
        fprintf('Completed %d / %d\n', i, num_ics);
    end
end

% Save the dataset
save('l96_udon.mat', 't', 'usol_all', 'u0_all');
disp('Data saved to l96_udon.mat');


function dudt = lorenz96(~, u, N, F)
    % Vectorized Lorenz '96 using circshift for periodic boundary conditions
    % dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    
    u_plus_1  = circshift(u, -1);
    u_minus_1 = circshift(u, 1);
    u_minus_2 = circshift(u, 2);
    
    dudt = (u_plus_1 - u_minus_2) .* u_minus_1 - u + F;
end