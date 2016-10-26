% This file conduct simulation studies for the Biometrics paper
% Compare SIPCA_A (general) SIPCA_B (orthogonal) 
% Focus on loading V estimates, low-rank structure recovery




%% simulation setting
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
Nsim=100;


for choosesetting=1:3
switch choosesetting
    case 1 % no supervision (JIVE setting, or exact SIPCA_A with B=0, or roughly SIPCA_B with B=0)
        simname='JIVE-Setting';
        temp=GramSchmidt(randn(n,r0+r1+r2));
        temp=bsxfun(@minus,temp,mean(temp,1));
        U0=temp(:,1:r0);
        U1=temp(:,(r0+1):(r0+r1));
        U2=temp(:,(r0+r1+1):end);
        X=GramSchmidt(randn(n,q),[U0,U1,U2]); % irrelevant covariates
        X=bsxfun(@minus,X,mean(X,1));
        B0=zeros(q,r0);
        B1=zeros(q,r1);
        B2=zeros(q,r2);
        V0=GramSchmidt(randn(p1+p2,r0));
        V1=GramSchmidt(randn(p1,r1));
        V2=GramSchmidt(randn(p2,r2));
        V_grand=[V0,blkdiag(V1,V2)];
        Joint=U0*diag([200,100])*V0';
        Jnt1=Joint(:,1:p1);
        Jnt2=Joint(:,(p1+1):end);
        Ind1=U1*diag([150,100,50])*V1';
        Ind2=U2*diag([120,90,60])*V2';
        %
        Data1_SNR=[norm(Jnt1,'fro'),norm(Ind1,'fro'),sigma1*sqrt(n*p1)]
        Data2_SNR=[norm(Jnt2,'fro'),norm(Ind2,'fro'),sigma2*sqrt(n*p2)]
        %
        lowrank=[Jnt1+Ind1,Jnt2+Ind2];
        
           
    case 2 % SIPCA_A setting
        simname='SIPCA-A-Setting';
        X=randn(n,q);
        X=bsxfun(@minus,X,mean(X,1));
        B0=hard_thres(randn(q,r0),0.3)*3; % partially sparse
        B1=hard_thres(randn(q,r1),0.3)*3; % partially sparse
        B2=hard_thres(randn(q,r2),0.3)*3; % partially sparse
        F0=bsxfun(@times,randn(n,r0),[9,5]);F0=bsxfun(@minus,F0,mean(F0,1));
        F1=bsxfun(@times,randn(n,r1),[6,4,2]);F1=bsxfun(@minus,F1,mean(F1,1));
        F2=bsxfun(@times,randn(n,r2),[7,3,1]);F2=bsxfun(@minus,F2,mean(F2,1));
        %
        U0_SNR=[norm(X*B0,'fro'),norm(F0,'fro')]
        U1_SNR=[norm(X*B1,'fro'),norm(F1,'fro')]
        U2_SNR=[norm(X*B2,'fro'),norm(F2,'fro')]
        %
        U0=X*B0+F0;
        U1=X*B1+F1;
        U2=X*B2+F2;
        V0=GramSchmidt([randn(p1,r0)*2;randn(p2,r0)]);
        V1=GramSchmidt(randn(p1,r1));
        V2=GramSchmidt(randn(p2,r2));
        V_grand=[V0,blkdiag(V1,V2)];        
        Joint=U0*V0';
        Jnt1=Joint(:,1:p1);
        Jnt2=Joint(:,(p1+1):end);
        Ind1=U1*V1';
        Ind2=U2*V2';
        %
        Data1_SNR=[norm(Jnt1,'fro'),norm(Ind1,'fro'),sigma1*sqrt(n*p1)]
        Data2_SNR=[norm(Jnt2,'fro'),norm(Ind2,'fro'),sigma2*sqrt(n*p2)]
        %
        lowrank=[Jnt1+Ind1,Jnt2+Ind2];
  
        
        
    case 3 % SIPCA_B setting
        simname='SIPCA-B-Setting';
        X=randn(n,q);
        X=bsxfun(@minus,X,mean(X,1));
        B0=hard_thres(randn(q,r0),0.3)*3; % partially sparse
        B1=hard_thres(randn(q,r1),0.3)*3; % partially sparse
        B2=hard_thres(randn(q,r2),0.3)*3; % partially sparse
        F0=bsxfun(@times,randn(n,r0),[9,5]);F0=bsxfun(@minus,F0,mean(F0,1));
        F1=bsxfun(@times,randn(n,r1),[6,4,2]);F1=bsxfun(@minus,F1,mean(F1,1));
        F2=bsxfun(@times,randn(n,r2),[7,3,1]);F2=bsxfun(@minus,F2,mean(F2,1));
        %
        U0_SNR=[norm(X*B0,'fro'),norm(F0,'fro')]
        U1_SNR=[norm(X*B1,'fro'),norm(F1,'fro')]
        U2_SNR=[norm(X*B2,'fro'),norm(F2,'fro')]
        %
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
        %
        Data1_SNR=[norm(Jnt1,'fro'),norm(Ind1,'fro'),sigma1*sqrt(n*p1)]
        Data2_SNR=[norm(Jnt2,'fro'),norm(Ind2,'fro'),sigma2*sqrt(n*p2)]
        %
        lowrank=[Jnt1+Ind1,Jnt2+Ind2];
        
   
end;
        


%% simulation run

rec_V0=zeros(Nsim,4); % Grassmannian dist
rec_V1=zeros(Nsim,4);
rec_V2=zeros(Nsim,4);
rec_V0max=zeros(Nsim,4); % Max Principal Angle
rec_V1max=zeros(Nsim,4);
rec_V2max=zeros(Nsim,4);
rec_Vallmax=zeros(Nsim,5); % Max Principal Angle for all
rec_lowrank=zeros(Nsim,5); % Frobenius norm of low-rank pattern difference


for nsim=1:Nsim
    disp(['sim run: ', num2str(nsim),'...']);
    
    % simulate data 
    E1=randn(n,p1)*sigma1;E1=bsxfun(@minus,E1,mean(E1,1));
    E2=randn(n,p2)*sigma2;E2=bsxfun(@minus,E2,mean(E2,1));
    Y1=Jnt1+Ind1+E1;
    Y2=Jnt2+Ind2+E2;    
    
   
    % SIFA_A 
    [B0_A, B_A, V0_A, Vind_A, se2_A, Sf0_A, Sf_A, U_A]=SIFA_A(X,{Y1,Y2},r0,r,...
        struct('sparsity',0,'Tol',0.01));
    V1_A=Vind_A{1};
    V2_A=Vind_A{2};
    V_grand_A=[V0_A,blkdiag(V1_A,V2_A)];
    lowrank_A=U_A*[V0_A,blkdiag(V1_A,V2_A)]';
      
    % SIFA_A sparse
    [B0_As, B_As, V0_As, Vind_As, se2_As, Sf0_As, Sf_As, U_As]=SIFA_A(X,{Y1,Y2},r0,r,...
        struct('sparsity',1,'Tol',0.01)); 
    V1_As=Vind_As{1};
    V2_As=Vind_As{2};
    V_grand_As=[V0_As,blkdiag(V1_As,V2_As)];
    lowrank_As=U_As*[V0_As,blkdiag(V1_As,V2_As)]';
    
    % SIFA_B
    [B0_B, B_B, V0_B, Vind_B, se2_B, Sf0_B, Sf_B, U_B]=SIFA_B(X,{Y1,Y2},r0,r,...
        struct('sparsity',0,'Tol',0.01));
    V1_B=Vind_B{1};
    V2_B=Vind_B{2};
    V_grand_B=[V0_B,blkdiag(V1_B,V2_B)];
    lowrank_B=U_B*[V0_B,blkdiag(V1_B,V2_B)]';
 
    % SIFA_B sparse
    [B0_Bs, B_Bs, V0_Bs, Vind_Bs, se2_Bs, Sf0_Bs, Sf_Bs, U_Bs]=SIFA_B(X,{Y1,Y2},r0,r,...
        struct('sparsity',1,'Tol',0.01));
    V1_Bs=Vind_Bs{1};
    V2_Bs=Vind_Bs{2};    
    V_grand_Bs=[V0_Bs,blkdiag(V1_Bs,V2_Bs)];
    lowrank_Bs=U_Bs*[V0_Bs,blkdiag(V1_Bs,V2_Bs)]';
    
    
    % PCA
    [U_SVD,D_SVD,V_grand_SVD]=svds([Y1,Y2],r0+r1+r2);
    lowrank_SVD=U_SVD*D_SVD*V_grand_SVD';
    
    
    
    
    
    
    
    % record 
    rec_V0(nsim,:)=[GrassDist(V0,V0_A),GrassDist(V0,V0_As),...
        GrassDist(V0,V0_B),GrassDist(V0,V0_Bs)];
    rec_V1(nsim,:)=[GrassDist(V1,V1_A),GrassDist(V1,V1_As),...
        GrassDist(V1,V1_B),GrassDist(V1,V1_Bs)];
    rec_V2(nsim,:)=[GrassDist(V2,V2_A),GrassDist(V2,V2_As),...
        GrassDist(V2,V2_B),GrassDist(V2,V2_Bs)];
    %
    rec_V0max(nsim,:)=[PrinAngle(V0,V0_A),PrinAngle(V0,V0_As),...
        PrinAngle(V0,V0_B),PrinAngle(V0,V0_Bs)];
    rec_V1max(nsim,:)=[PrinAngle(V1,V1_A),PrinAngle(V1,V1_As),...
        PrinAngle(V1,V1_B),PrinAngle(V1,V1_Bs)];
    rec_V2max(nsim,:)=[PrinAngle(V2,V2_A),PrinAngle(V2,V2_As),...
        PrinAngle(V2,V2_B),PrinAngle(V2,V2_Bs)];
    %
    rec_Vallmax(nsim,:)=[PrinAngle(V_grand,V_grand_SVD),...
        PrinAngle(V_grand,V_grand_A),PrinAngle(V_grand,V_grand_As),...
        PrinAngle(V_grand,V_grand_B),PrinAngle(V_grand,V_grand_Bs)];
    %
    rec_lowrank(nsim,:)=[norm(lowrank-lowrank_SVD,'fro'),...
        norm(lowrank-lowrank_A,'fro'),norm(lowrank-lowrank_As,'fro'),...
        norm(lowrank-lowrank_B,'fro'),norm(lowrank-lowrank_Bs,'fro')];
end;


%% compare results
figure(1);clf;
subplot(3,2,1);
boxplot(rec_V0,{'SIFA_A','SIPCA_As','SIPCA_B','SIPCA_Bs'});
ylabel('V0 Grassmannian Loss','fontsize',15); 
title([simname,' V0 Comparison'],'fontsize',15);
subplot(3,2,2);
boxplot(rec_V0max,{'SIPCA_A','SIPCA_As','SIPCA_B','SIPCA_Bs'});
ylabel('V0 Max Prin Angle','fontsize',15); 
title([simname,' V0 Comparison'],'fontsize',15);
%
subplot(3,2,3);
boxplot(rec_V1,{'SIPCA_A','SIPCA_As','SIPCA_B','SIPCA_Bs'});
ylabel('V1 Grassmannian Loss','fontsize',15); 
title([simname,' V1 Comparison'],'fontsize',15);
subplot(3,2,4);
boxplot(rec_V1max,{'SIPCA_A','SIPCA_As','SIPCA_B','SIPCA_Bs'});
ylabel('V1 Max Prin Angle','fontsize',15); 
title([simname,' V1 Comparison'],'fontsize',15);
%
subplot(3,2,5);
boxplot(rec_V2,{'SIPCA_A','SIPCA_As','SIPCA_B','SIPCA_Bs'});
ylabel('V2 Grassmannian Loss','fontsize',15); 
title([simname,' V2 Comparison'],'fontsize',15);
subplot(3,2,6);
boxplot(rec_V2max,{'SIPCA_A','SIPCA_As','SIPCA_B','SIPCA_Bs'});
ylabel('V2 Max Prin Angle','fontsize',15); 
title([simname,' V2 Comparison'],'fontsize',15);




figure(2);clf;
subplot(2,1,1);
boxplot(rec_Vallmax,{'SVD','SIPCA_A','SIPCA_As','SIPCA_B','SIPCA_Bs'});
ylabel('V_all Max Prin Angle','fontsize',15); 
title([simname,' All V Space'],'fontsize',15);
%
subplot(2,1,2);
boxplot(rec_lowrank,{'SVD','SIPCA_A','SIPCA_As','SIPCA_B','SIPCA_Bs'});
ylabel('Low-Rank Structure Frobenius Loss','fontsize',15); 
title([simname,' Low-Rank Comparison'],'fontsize',15);

end;


