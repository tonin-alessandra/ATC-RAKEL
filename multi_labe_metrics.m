function [Absolute_false,Coverage,Absolute_true,Aiming,Accuracy]=multi_labe_metrics(Pre_Labels,test_target)
% output five multi-label Metrics
% multi_labe_metrics takes,
%       Pre_Labels   -  Multi-label predicted by ML-GKR, A QxM2 array, if the ith testing 
%                          instance belongs to the jth class, test_target(j,i) equals +1, 
%                           otherwise test_target(j,i) equals -1
%       test_target   -    A QxM2 array, if the ith testing instance belongs to the jth class, 
%                          test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%     
% and returns,
%       Absolute_false - refer to [1] for detailed description
%       Coverage - refer to [1] for detailed description
%       Absolute_true - refer to [1] for detailed description
%       Aiming - refer to [1] for detailed description
%       Accuracy - refer to [1] for detailed description
%[1] Cheng X, Zhao SG, Xiao X, et al. iATC-mISF: a multi-label classifier 
%     for predicting the classes of anatomical therapeutic chemicals.[J]. 
%     Bioinformatics (Oxford, England), 2016.

%compare Pre_Labels and test_target,ACC
res=Pre_Labels-test_target;
res=abs(res);
res=res./2;
[leiNum,instance_num]=size(Pre_Labels);
% Absolute true
acc=0;
for i=1:instance_num
    if(0==sum(res(:,i)))
        acc=acc+1;
    end
end
Absolute_true=acc/instance_num;

Lab=test_target+ones(leiNum,instance_num);
Lab=Lab./2;
preLab=Pre_Labels+ones(leiNum,instance_num);
preLab=preLab./2;

% Accuracy
fenzi=preLab&Lab;
fenmu=preLab|Lab;

mlACC=0;
for i=1:instance_num
    mlACC=mlACC+sum(fenzi(:,i))/sum(fenmu(:,i));
end
Accuracy=mlACC/instance_num;

fenzi=preLab&Lab;
fenmu=preLab;
mlPRE_mu=0;
mlPRE_zi=0;
for i=1:leiNum
    mlPRE_zi=mlPRE_zi+sum(fenzi(i,:));
    mlPRE_mu=mlPRE_mu+sum(fenmu(i,:));
end
% Aiming,mlPRE
Aiming=mlPRE_zi/mlPRE_mu;


fenzi=preLab&Lab;
fenmu=Lab;
mlREC_zi=0;
mlREC_mu=0;
for i=1:leiNum
    mlREC_zi=mlREC_zi+sum(fenzi(i,:));
    mlREC_mu=mlREC_mu+sum(fenmu(i,:));
end

Coverage=mlREC_zi/mlREC_mu;

%absolute false
prefenzi_1=sum(preLab,1);
prefenzi_2=sum(Lab,1);
abFalse=0;
for i=1:instance_num
    fen=abs(prefenzi_1(i)-prefenzi_2(i))/leiNum;
    abFalse=abFalse+fen;
end
Absolute_false=abFalse/instance_num;
