clc;
clear;
addpath(genpath('.'));
load('birds.mat');
warning off;

% birds,corel5k
target = targets';

% emotions,scene,health,yeast,image,enron
% data=dataset;
% target=class;
% target(target==-1)=0;

% t0 = clock;
if exist('train_data','var')==1
    data=[train_data;test_data];
    target=[train_target,test_target];
end

data = double(data);
num_class = size(target,1);
num_data = size(data,1);

%different numbers of features
selectedFN = 10;  

randorder = randperm(num_data);
cv_num = 5;

Result_All = cell(1,selectedFN);
for i=1:selectedFN
    Result_All{i} = zeros(15,5);
end

% parameters
para.alpha = 1; para.beta = 1; para.gamma = 1;

Num = 10;Smooth = 1; 

for j = 1:5
    
    fprintf('Running Fold - %d/%d \n',j,cv_num);
    
    [cv_train_data,cv_train_target,cv_test_data,cv_test_target ] = generateCVSet(data,target',randorder,j,cv_num);
    cv_train_target=cv_train_target';
    cv_test_target=cv_test_target';
    
    tmp_cv_train_target = cv_train_target;
    tmp_cv_train_target(tmp_cv_train_target==0) = -1;
    tmp_cv_test_target = cv_test_target;
    tmp_cv_test_target(tmp_cv_test_target==0) = -1;
    
    t0 = clock;
    [ W, obj ] = MRDM( cv_train_data, cv_train_target', para );
%     display(etime(clock, t0));
%     plot(obj);

    [dumb , idx] = sort(sum(W.*W,2),'descend'); 
    
    for FeaNum=10:10:10*selectedFN
        
        fea = idx(1:FeaNum);
        
        % ML-KNN classifier
        [Prior,PriorN,Cond,CondN]=MLKNN_train(cv_train_data(:,fea),tmp_cv_train_target,Num,Smooth);
    
        [HammingLoss,RankingLoss,Coverage,OneError,Average_Precision,macrof1,microf1,EBA,EBP,EBR,EBF,LBA,LBP,LBR,LBF]=...
            MLKNN_test(cv_train_data(:,fea),tmp_cv_train_target,cv_test_data(:,fea),tmp_cv_test_target,Num,Prior,PriorN,Cond,CondN);
        
        Result_All{FeaNum/10}(1,j)=HammingLoss;
        Result_All{FeaNum/10}(2,j)=RankingLoss;
        Result_All{FeaNum/10}(3,j)=Coverage;
        Result_All{FeaNum/10}(4,j)=OneError;
        Result_All{FeaNum/10}(5,j)=Average_Precision;
        Result_All{FeaNum/10}(6,j)=macrof1;
        Result_All{FeaNum/10}(7,j)=microf1;
        Result_All{FeaNum/10}(8,j)=EBA;
        Result_All{FeaNum/10}(9,j)=EBP;
        Result_All{FeaNum/10}(10,j)=EBR;
        Result_All{FeaNum/10}(11,j)=EBF;
        Result_All{FeaNum/10}(12,j)=LBA;
        Result_All{FeaNum/10}(13,j)=LBP;
        Result_All{FeaNum/10}(14,j)=LBR;
        Result_All{FeaNum/10}(15,j)=LBF;
        
    end
% 	display(etime(clock, t0)/60);
end

% Experiment results
Avg_Result = cell(1,selectedFN);

for i=1:selectedFN
    Avg_Result{i} = zeros(15,2);
end

for i=1:selectedFN
    Avg_Result{i}(:,1)=mean(Result_All{i},2);
    Avg_Result{i}(:,2)=std(Result_All{i},1,2);
end

% time = etime(clock, t0);
% display(time/60);
% plot(obj);

