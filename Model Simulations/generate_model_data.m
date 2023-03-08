function generate_model_data()
%generate_model_data genrates a population vector time-series for a model
%simulation of E-I neurons

%  Author: Keshav Srinivasan, March 2023

%% Parameters
N = 1e6; % Number of neurons
steps = 1e8; % Number of timesteps (Can be parallelized to doing smaller runs, for e.g. 100 runs of 1e6 each. Note: this would have 100 "edges" but these would not affect statistics)
i_vec = 5:5:N; % Every 5th neuron is inhibitory (i.e: E-I ratio 4:1)
e_vec = setdiff(1:N,i_vec); % All other neurons are excitatory
drive = 20/N; % Poisson drive (~20 neurons per timestep on average are activated)
J = 10; % Model parameter (See main text)
g = 3.5; % E-I balance parameter. (g = 3.5 is critical)

%% Initialize subsamples 
% Pick fraction of x random neurons from the whole network
sub_10 = randperm(N,0.1*N);
sub_1 = randperm(N,0.01*N);
sub_01 = randperm(N,0.001*N);
sub_001 = randperm(N,0.0001*N);

m = matfile('Insert file_name here'); % use this file to save progress from time to time
X = zeros(N,1);
pop_vec_full = zeros(steps,1);
pop_vec_sub_10 = zeros(steps,1);
pop_vec_sub_1 = zeros(steps,1);
pop_vec_sub_01 = zeros(steps,1);
pop_vec_sub_001 = zeros(steps,1);
for t = 1:steps
    if (mod(t,steps/10) == 0) % Every 10% display progress and save variables
        disp(t*10/steps);
        m.full = pop_vec_full;
        m.sub_10 = pop_vec_sub_10;
        m.sub_1 = pop_vec_sub_1;
        m.sub_01 = pop_vec_sub_01;
        m.sub_001 = pop_vec_sub_001;
    end  
    add = (rand(N,1) < drive); % add poisson drive
    X = ((X + add) >0);
    V = ((J/N)*sum(X(e_vec)) - (g*J/N)*sum(X(i_vec))).*(1-X); % Run evolution of membrane potential
    X = (rand(N,1) < V); % Probablistic firing 
    pop_vec_full(t) = sum(X);
    pop_vec_sub_10(t) = sum(X(sub_10));
    pop_vec_sub_1(t) = sum(X(sub_1));
    pop_vec_sub_01(t) = sum(X(sub_01));
    pop_vec_sub_001(t) = sum(X(sub_001));
end

% Final saves
m.full = pop_vec_full;
m.sub_10 = pop_vec_sub_10;
m.sub_1 = pop_vec_sub_1;
m.sub_01 = pop_vec_sub_01;
m.sub_001 = pop_vec_sub_001;
end
    

