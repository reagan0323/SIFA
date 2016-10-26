% This file conduct simulation studies for rank estimation (Biometrics revision)
% Implement likelihood-CV idea to search for r0,r1,r2 from a small candidate set
%
% 9/12/2016 by Gen Li

%% Simulate data
% generic setting
K=2;
n=500; 
p1=200;
p2=200;
q=10;
p=[p1,p2];
r0=2; 
r1=3;
r2=3;
r=[r1,r2];
sigma1=2; % std
sigma2=3;



simname='SIPCA-B-Setting'; % same as simulation in paper
X=randn(n,q);
X=bsxfun(@minus,X,mean(X,1));
B0=hard_thres(randn(q,r0),0.3)*3; % partially sparse
B1=hard_thres(randn(q,r1),0.3)*3; % partially sparse
B2=hard_thres(randn(q,r2),0.3)*3; % partially sparse
F0=bsxfun(@times,randn(n,r0),[9,5]);F0=bsxfun(@minus,F0,mean(F0,1));
F1=bsxfun(@times,randn(n,r1),[6,4,2]);F1=bsxfun(@minus,F1,mean(F1,1));
F2=bsxfun(@times,randn(n,r2),[7,3,1]);F2=bsxfun(@minus,F2,mean(F2,1));
U0=X*B0+F0;
U1=X*B1+F1;
U2=X*B2+F2;
temp1=GramSchmidt(randn(p1,r0+r1));
temp2=GramSchmidt(randn(p1,r0+r2));
V01=temp1(:,1:r0)*(1/sqrt(K));
V02=temp2(:,1:r0)*(1/sqrt(K));
V0=[V01;V02];
V1=temp1(:,(r0+1):end);
V2=temp2(:,(r0+1):end);
V_grand=[V0,blkdiag(V1,V2)];
Joint=U0*V0';
Jnt1=Joint(:,1:p1);
Jnt2=Joint(:,(p1+1):end);
Ind1=U1*V1';
Ind2=U2*V2';
E1=randn(n,p1)*sigma1;E1=bsxfun(@minus,E1,mean(E1,1));
E2=randn(n,p2)*sigma2;E2=bsxfun(@minus,E2,mean(E2,1));
Y1=Jnt1+Ind1+E1;
Y2=Jnt2+Ind2+E2;
      


%% LCV method

% setup sample groups 
Nfold=10; % must be a devisor of n
rng(1234)
randorder=randsample(n,n);
cutoff=(0:Nfold)*n/Nfold;
% candidate tuning grids
rcand=[1,2,2
    2,2,2;
    1,3,3;
    3,2,2;
    2,3,3;
    3,3,3;
    3,4,3;
    3,4,4;
    4,4,4]; % each row is a candidate, [r0,r1,r2]
LCV_score=zeros(Nfold,size(rcand,1)); % LCV scores for each CV run on each candidate 
clc
disp([num2str(Nfold),'-fold LCV for ',num2str(size(rcand,1)),' candidates.'])

for ifold=1:Nfold
    disp(['Running Fold: ',num2str(ifold)])
    X_train=X; 
    X_train(cutoff(ifold)+1:cutoff(ifold+1),:)=[];
    X_test=X(cutoff(ifold)+1:cutoff(ifold+1),:);
    Y1_train=Y1;
    Y1_train(cutoff(ifold)+1:cutoff(ifold+1),:)=[];
    Y1_test=Y1(cutoff(ifold)+1:cutoff(ifold+1),:);
    Y2_train=Y2;
    Y2_train(cutoff(ifold)+1:cutoff(ifold+1),:)=[];
    Y2_test=Y2(cutoff(ifold)+1:cutoff(ifold+1),:);
    
    for icand=1:size(rcand,1)
        r0_test=rcand(icand,1);
        r_test=rcand(icand,2:3);
        % fit model on training data
        [B0_est, B_est, V0_est, Vind_est, se2_est, Sf0_est, Sf_est, U_est]=...
            SIPCA_B(X_train,{Y1_train,Y2_train},r0_test,r_test,struct('sparsity',0,'Tol',0.01));
        % evaluation likelihood on test data 
        % LCV score is negative loglik, the smaller the better
        LCV_score(ifold,icand)=-loglikelihood(X_test,{Y1_test,Y2_test},B0_est,B_est,V0_est, Vind_est,se2_est,Sf0_est,Sf_est);
    end;
end;


% plot CV results
figure(1);clf
plot(LCV_score','ko--','linewidth',1.5);
hold on;
plot(mean(LCV_score,1),'r*-','markersize',20,'linewidth',2)
xlabel('Candidate Rank Sets','fontsize',15);
ylabel('LCV Scores','fontsize',15);
title('10-Fold Likelihood Cross Validation','fontsize',20)
set(gca,'fontsize',15,'xtick',[1:9],'xticklabel',...
    {'(1,2,2)','(2,2,2)','(1,3,3)','(3,2,2)','(2,3,3)','(3,3,3)','(3,4,3)','(3,4,4)','(4,4,4)'});
