% Parameters
sigma = 10;
beta  = 8/3;
rho   = 28;

% Narrower time domain for DeepONet window training
tspan = [0, 1.0]; 
num_points = 101; 
t = linspace(tspan(1), tspan(2), num_points);

% Number of Initial Conditions to sample
num_ics = 500; 

% Preallocate arrays to store the dataset
usol_all = zeros(num_ics, num_points, 3);
u0_all = zeros(num_ics, 3);

% Eq system
f = @(t, u) [ ...
    sigma * (u(2) - u(1)); ...
    u(1) * (rho - u(3)) - u(2); ...
    u(1) * u(2) - beta * u(3) ...
];

opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-14);

disp('Generating trajectories...');
for i = 1:num_ics
    % Randomly sample initial conditions within the general L63 attractor bounds
    % x in [-20, 20], y in [-30, 30], z in [0, 50]
    u0 = [ 
        -20 + 40 * rand(); 
        -30 + 60 * rand(); 
          0 + 50 * rand() 
    ];
    
    u0_all(i, :) = u0';
    
    % Solve directly on the target time points
    [~, u] = ode113(f, t, u0, opts);
    
    usol_all(i, :, :) = u;
    
    if mod(i, 50) == 0
        fprintf('Completed %d / %d\n', i, num_ics);
    end
end

% Save the dataset
save('l63_udon.mat', 't', 'usol_all', 'u0_all');
disp('Data saved to l63_udon.mat');