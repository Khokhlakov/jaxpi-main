% Parameters
sigma = 10;
beta  = 8/3;
rho   = 28;

tspan = [0, 25];
u0    = [1; 1; 1]; 

% Eq system
f = @(t, u) [ ...
    sigma * (u(2) - u(1)); ...
    u(1) * (rho - u(3)) - u(2); ...
    u(1) * u(2) - beta * u(3) ...
];

opts = odeset('RelTol', 1e-13, 'AbsTol', 1e-14);

% Solve
[T, u] = chebfun.ode113(f, tspan, u0, opts);

% ndividual chebfuns for x, y, z
x_cheb = u(:, 1);
y_cheb = u(:, 2);
z_cheb = u(:, 3);

% Sample
num_points = 2501; 
t = linspace(tspan(1), tspan(2), num_points);

xsol = x_cheb(t);
ysol = y_cheb(t);
zsol = z_cheb(t);

save('l63.mat', 't', 'xsol', 'ysol', 'zsol', '-v7.3');