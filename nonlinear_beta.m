Ret = clean_CRSP(Table,19950101,20190101);
Rm = clean_CRSP(Rm19602019,19950101,20190101,'market');
clc
%% Step 1: estimate the betas from the time series regression
beta = mat_regress(Ret, Rm);
N = size(Ret,2);
T = size(Ret,1);

%% 1.5 see the rolling window estimate of betas
% for t = 1:6
%     temp = mat_regress(Ret(1+(t-1)*1000: t*1000,:), Rm(1+(t-1)*1000: t*1000,:));
%     beta_rolling(t,:) = temp(2,:);
% end
% %% 
% plot(beta_rolling(:,6))
% it seems that depending on the frequency we re-calculate betas, it can vary a lot. but 
% if we use 5 years data, then it looks better.

%% Step 2: the cross-sectional regression
Ret_mean = mean(Ret);
Ret_var = var(Ret);
scatter(beta(2,:),Ret_mean);
% hold on 
% scatter(beta(2,:),Ret_var);
gamma = fitglm(beta(2,:),Ret_mean');
gamma

% hold off
% ksrlin(beta(2,:),Ret_mean,0.2)

%% Step 3: Now we split the sample into portfolios and windows
global window group 
window = create_window_matrix(Ret, 1000, 100);
% 1. calculate beta; 2. sort returns ascendingly according to beta
% 3. group the returns into portfolio based on group_size
init_beta = find_beta(Ret, Rm, 1);
group_size = 1;
Rp_index = portfolio_index(init_beta(2,:)');
Ret = Ret(:,Rp_index);
group = create_window_matrix(Ret', group_size, group_size)';
clear init_beta  Rp_index %group_size

%% Step 4: compute the betas and find portfolio beta
beta = zeros(size(window,2),size(Ret,2));
for t = 1 : size(window,2)
	temp_beta = find_beta(Ret, Rm, t);
	beta(t,:) = temp_beta(2,:);
end
portfolio_beta = beta * group'/group_size;
clear temp_beta

for j = 100:105
    plot(1:size(window,2),beta(:,j));
    hold on 
end
hold off




%% Step 5: cross-section with portfolio return and beta
for t = 1:size(window,2)
    Rp_mean(:,t) =(Ret(in_window(t),:)*group')/group_size;
    figure;
    scatter(portfolio_beta(:),Rp_mean(:,t))

end
    
%% Local Functions
function beta = mat_regress(Y, X)
	X = [ones(size(X,1),1),X];
	P = X'*X\X';
	beta = P * Y;
end

function logic = in_window(t)
	global window
	logic = window(:,t) == 1;
end

function logic = in_group(n)
	global group
	logic = group(n,:) == 1;
end
function beta = find_beta(Ret, Rm, t)
	beta = mat_regress(Ret(in_window(t),:) , Rm(in_window(t), :));
end

%% Put beta as a column_vector
function index = portfolio_index(sorting_column)
	k = [(1:length(sorting_column))', sorting_column];
	k = sortrows(k,2);
	index = k(:,1)';
end