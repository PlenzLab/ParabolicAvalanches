% Script to extract pdf curves for S, T and <S> vs T

%   Author: Keshav Srinivasan, March 2023

%% Parameters
tsteps = 1e8; % Number of time steps of the time-series vector
d = 25; % Range of k (time-bin) 

%Initalize variables
S = zeros(tsteps/10,d); 
T = zeros(tsteps/10,d);
counts = zeros(1,d);
p = 1; % The subsampling level (p = 1 is 100% sampled, p = 2 is 10%....)

%% Main Loop
for k = 1:25
    disp(k);
    % See file from the data repository
    % https://doi.org/10.5281/zenodo.7703224 for an example dataset
    total_vec = data(p,:); % Set pop_vec
    threshold = thresholds(p,k); % Set threshold
    total_vec(total_vec<=threshold)=0; % Threshold the pop_vec
    flag = 0;
    count = 1;
    S_l = zeros(tsteps/10,1);
    T_l = zeros(tsteps/10,1);
    delay = k; %bin size
    binned_vec = zeros(floor(tsteps/delay),1);
    for i = 1:floor(tsteps/delay)
        binned_vec(i) = sum(total_vec((i-1)*delay+1:i*delay));
        if flag == 0 && binned_vec(i) > 0
            t = i;
            flag = 1;
        end
        if flag == 1 && binned_vec(i) == 0
            S_l(count) = sum(binned_vec(t:i));
            T_l(count) = i - t;
            count = count + 1;
            flag = 0;
        end
    end
    S(:,k) = S_l;
    T(:,k) = T_l;
    counts(:,k) = count;
end

%% Plotting
colors = colormap(winter(d));
M.S = S;
M.T = T;
Tmax = 2e4;
Ts = 1:Tmax; % Range of durations
Final = Ts;
avgS = zeros(d,Tmax); % <S> for a particular T
for i = 1:d
    figure(3); % <S> vs T
    totS = zeros(1,Tmax);
    numS = zeros(1,Tmax);
    S_temp = M.S(:,i);
    T_temp = M.T(:,i);
    for t = 1:100000000
        if(T_temp(t) == 0) % End of avalanches
            break;
        end
        if (T_temp(t)<length(Ts))
            totS(T_temp(t)) = totS(T_temp(t)) + S_temp(t);
            numS(T_temp(t)) = numS(T_temp(t)) + 1;
        end
 
    end
    avgS(i,:) = totS./numS;
    loglog(Ts(~isnan(avgS(i,:))),avgS(i,~isnan(avgS(i,:))),'LineWidth',2,'Color',colors(i,:)); hold on

    figure(2); % S pdf
    s_vector = space(50,max(S_temp),1,threshold);
    s_pdf = histcounts(S_temp,s_vector,'Normalization','countdensity');
    s_pdf = s_pdf./nnz(S_temp);
    loglog(s_vector(1:end-1),s_pdf,'LineWidth',1,'Color',colors(i,:)); hold on

    figure(1); % T pdf
    t_vector = space(30,max(T_temp),1,1);
    t_pdf = histcounts(T_temp,t_vector,'Normalization','countdensity');
    t_pdf = t_pdf./nnz(T_temp);
    loglog(t_vector(1:end-1),t_pdf,'LineWidth',1,'Color',colors(i,:)); hold on
end

%% Adding guide lines and labels and limits
figure(1)
    plot((1:1e3),1*(1:1e3).^-2,'--k','Linewidth',2); 
    xlabel('T');
    ylabel('P(T)');
    xlim([1,1e3]);
    set(gca,'Fontsize',14);

figure(2)
    plot((1:1e7),1*(1:1e7).^-1.5,'--k','Linewidth',2);
    xlabel('S');
    ylabel('P(S)');
    xlim([1e2,1e7]);
    c = lines;
    set(gca,'Fontsize',14);

figure(3)
    plot(Ts,2*Ts.^2,'--k','Linewidth',2); hold on
    plot(Ts,1.5.*Ts.^1.5,'--','Linewidth',2,'Color',c(1,:)); hold on
    plot(Ts,200.*Ts.^1,'--','Linewidth',2,'Color',c(2,:)); hold on
    xlim([1,1e3]);
    ylim([5e1,1e7]);
    xlabel('T');
    ylabel('<S>');
    set(gca,'Fontsize',14);


  
        