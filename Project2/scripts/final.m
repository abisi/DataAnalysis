% Comparing different methods


clear all;
close all;
load('../data/Data.mat');

%% Partition 70-30, standardization, PCA

proportion = 0.7;     
rows = size(Data,1); %12862
sep_idx = round(rows*proportion);
train = Data(1:sep_idx,:);
test = Data(sep_idx:end,:); %here we keep order because we wanna predict future values based on past values

%Stdize
[std_train, mu, sigma] = zscore(train); 
std_test = (test - mu ) ./ sigma; %using same normalization coefficients 

%PCA
[coeff, score, latent] = pca(std_train);
pca_train = std_train * coeff;
pca_test = std_test * coeff; %using same pca coefficients

%Choose PCs (+ graphs)
cum_Var=cumsum(latent)/sum(latent);
num_PC=1:length(latent);
figure;
plot(num_PC, cum_Var, 'r'); hold on;
xlabel('Number of PCs');
ylabel('Percentage of the total variance');
title('Total information carried by Principal Components');

idx90=find(cum_Var>0.9);
pc90=num_PC(idx90(1));
threshold90=line([pc90 pc90], [0 1]);
set(threshold90,'LineWidth',2,'color','blue');

figure
bar(latent);

%% Targets

%Train
target_posx = PosX(1:sep_idx); % x position reg. targets
target_posy = PosY(1:sep_idx); % y position reg. targets
%Test
target_posx_te = PosX(sep_idx:end); 
target_posy_te = PosY(sep_idx:end);

%% Initialization of errors
degrees = [1:10];

mse_poly_x = zeros(length(degrees),length(num_PC));
mse_lasso_x = zeros(length(degrees),length(num_PC));
mse_elastic_x = zeros(length(degrees),length(num_PC));

mse_poly_y = zeros(length(degrees),length(num_PC));
mse_lasso_y = zeros(length(degrees),length(num_PC));
mse_elastic_y = zeros(length(degrees),length(num_PC));

mse_poly_x_te = zeros(length(degrees),length(num_PC));
mse_lasso_x_te = zeros(length(degrees),length(num_PC));
mse_elastic_x_te = zeros(length(degrees),length(num_PC));

mse_poly_y_te = zeros(length(degrees),length(num_PC));
mse_lasso_y_te = zeros(length(degrees),length(num_PC));
mse_elastic_y_te = zeros(length(degrees),length(num_PC));

%% Polynomial regression - Grid Search

FM_train = pca_train;
I_train = ones(size(FM_train,1),1);
X_train = [I_train FM_train];
FM_test = pca_test;
I_test = ones(size(FM_test,1),1);
X_test = [I_test FM_test];

for degree_idx=1:1:length(degrees)
    X_train = [X_train  FM_train.^(degree_idx)];
    X_test = [X_test  FM_test.^(degree_idx)];
    
    for PC_idx=1:10:length(num_PC)
        counter=PC_idx+1; %why ?
        %regres
        bx = regress(target_posx,X_train(:,1:counter));
        by = regress(target_posy,X_train(:,1:counter));
        %predict
        x_hat = X_train(:,1:counter) * bx; 
        y_hat = X_train(:,1:counter) * by;
        x_hat_te = X_test(:,1:counter) * bx; 
        y_hat_te = X_test(:,1:counter) * by;
        %training error
        mse_poly_x(degree_idx, PC_idx) = immse(target_posx, x_hat);
        mse_poly_y(degree_idx, PC_idx) = immse(target_posy, y_hat);
        %testing error
        mse_poly_x_te(degree_idx, PC_idx) = immse(target_posx_te, x_hat_te);
        mse_poly_y_te(degree_idx, PC_idx) = immse(target_posy_te, y_hat_te);
    end
end

%% Let's plot the errors with increasing number of PCs
x = num_PC; %number PCs

figure;
subplot(1,2,1) 
s1 = surf(x, degrees, mse_poly_x(degrees,x)); hold on
s1_te = surf(x, degrees, mse_poly_x_te(degrees,x)); 
%s1.EdgeColor = 'none';
%s1_te.EdgeColor = 'none';
xlabel('Number of PCs');
ylabel('Polynomial degree');
zlabel('MSE');
title('Polynomial regression of PosX ');
legend('Training error');

subplot(1,2,2)
s2 = surf(x, degrees, mse_poly_x(degrees,x)); hold on
s2_te = surf(x, degrees, mse_poly_x_te(degrees,x));  
%s2.EdgeColor = 'none';
%s2_te.EdgeColor = 'none';
xlabel('Number of PCs');
ylabel('Polynomial degree');
zlabel('MSE');
title('Polynomial regression of PosY');
legend('Training error');

    

