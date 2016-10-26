function [B0,B,V_joint, V_ind,se2,Sf0,Sf,EU]=SIFA_B(X,Y,r0,r,paramstruct)
% This function fits SIPCA under the specific conditions.
% 1. V0k'V0k=(1/K)I, 
% 2. V0k'*Vk=0
% It assumes a linear relation, and may or may not impose sparsity on B
% Input data should have roughly equal scale (i.e., ||Y1||~||Y2||) because
% of the norm constraint on V_0k. 
%
% Input: 
%       X           n*q matrix, centered covariate data 
%       Y           1*K cell array, each cell is a n*pi centered primary
%                   data, should roughly contribute equally to joint struct
%       r0          scalar, prespecified rank of common structure
%       r           1*K vector, prespecified rank of specific structures
%       paramstruct
%            sparsity    1 (default), when est B0 or B, use LASSO with BIC
%                           to select the best tuning, suitable for high dimension
%                        0, no sparsity, only valid for low dimension
%            Niter       default 500, max number of iteration
%            Tol         default 1e-3, 
%
%
% Output:
%       B0          q*r0 matrix, coefficient for joint structure (may be sparse)
%       B           1*K cell array, each is a q*ri coefficient matrix (may be sparse)
%       V_joint     sum(p)*r0 matrix, stacked joint loadings, with orthonormal
%                   columns. 
%       V_ind       1*K array, each is a pi*ri loading matrix, with
%                   orthonormal columns
%       se2         1*K vector, noise variance for each phenotypic data set
%       Sf0         r0*r0 matrix, diagonal covariance matrix
%       Sf          1*K array, each is a ri*ri diagonal covariance matrix
%       EU          n*(r0+sum(ri)) matrix, conditional expectation of joint and individual scores
%
%
% Created: 2016.3.8
% By: Gen Li
% Modified: 2016.3.10:  change stopping rule from loglik diff to PrinAngle diff


sparsity=1;
max_niter=500;
convg_thres=1E-3; 
if nargin > 4 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'sparsity');
      sparsity=getfield(paramstruct,'sparsity');
  end;
  if isfield(paramstruct,'Niter');
      max_niter=getfield(paramstruct,'Niter');
  end;
  if isfield(paramstruct,'Tol');
      convg_thres=getfield(paramstruct,'Tol');
  end;
end;



% check dimension
K=length(Y);
p=zeros(1,K); % collect number of variables in each phenotypic data set
[n,q]=size(X);
if n<=q && ~sparsity
    warning('Must use sparse estimation to avoid overfitting!!!');
    sparsity=1;
end;
for k=1:K
    [n_temp,p(k)]=size(Y{k});
    if n_temp~=n
        error('Sample mismatch!');
    end;
end;
if sum((r0+r)>min(n,p))
    error('Too greedy on ranks!');
end;




% initial estimate
grandY=[];
for k=1:K
    grandY=[grandY,Y{k}];
end;
[U_joint_ini,D_joint_ini,V_joint_ini]=svds(grandY,r0); % note: segments of V_joint_ini corresponding to each data set may not be orthogonal as desired
U_joint_ini=U_joint_ini*D_joint_ini;
B0=zeros(q,r0);
if sparsity
    for i=1:r0
%         tic
        % Attention: lasso fcn always center X, Y, and coefficients are
        % calculated based on centered X and Y. The function will return a
        % separate column for intercept; if we center data outside the
        % function, the intercept will be nearly 0;
        [SpParam,FitInfo]=lasso(X,U_joint_ini(:,i),'LambdaRatio',0,'Standardize',true); 
        BIC_score=n*log(FitInfo.MSE)+log(n)*FitInfo.DF;
        [~,ind]=min(BIC_score);
        B0(:,i)=SpParam(:,ind);
%         toc
    end;
else % if sparsity=0, no B-sparsity
    B0=(X'*X)\X'*U_joint_ini;
end;
Sf0=diag(std(U_joint_ini-X*B0).^2);
V_ind={};
B={};
Sf={};
for k=1:K
    loc3=sum(p(1:(k-1)))+1;
    loc4=sum(p(1:k));
    Ycurrent=Y{k}-U_joint_ini*V_joint_ini(loc3:loc4,:)';
    [U_k_ini,D_k_ini,V_k_ini]=svds(Ycurrent,r(k));
    U_k_ini=U_k_ini*D_k_ini;
    V_ind{k}=V_k_ini;
    if sparsity
        for i=1:r(k)
            [SpParam,FitInfo]=lasso(X,U_k_ini(:,i),'LambdaRatio',0,'Standardize',true); 
            BIC_score=n*log(FitInfo.MSE)+log(n)*FitInfo.DF;
            [~,ind]=min(BIC_score);
            B{k}(:,i)=SpParam(:,ind);
        end;
    else % if sparsity=0, no B-sparsity
        B{k}=(X'*X)\X'*U_k_ini;
    end;
    Sf{k}=diag(std(U_k_ini-X*B{k}).^2);
    se2(k)=norm(Ycurrent-U_k_ini*V_k_ini','fro')^2/(n*p(k));
end;
% postprocess V_joint_ini such that it follows our identifiability condition
V_joint=zeros(sum(p),r0);
for k=1:K
    loc3=sum(p(1:(k-1)))+1;
    loc4=sum(p(1:k));
    V_joint_k=GramSchmidt(V_joint_ini(loc3:loc4,:),V_ind{k});
    V_joint(loc3:loc4,:)=V_joint_k*(1/sqrt(K));
end;
grandV=[];
for k=1:K
    grandV=blkdiag(grandV,V_ind{k});
end;
grandV=[V_joint,grandV];% sum(p)*(r0+sum(r))





% disp('Initial est done!')
loglik=loglikelihood(X,Y,B0,B,V_joint, V_ind,se2,Sf0,Sf);
recloglik=loglik;
maxloglik=loglik;


niter=0; 
diff=inf; 
diff_max=inf; 
tol_thres=-10;
recdiff=[];

recdiff_grandV=[];
recdiff_V_joint=[];
recdiff_V1=[];
recdiff_V2=[];
while (niter<=max_niter && abs(diff)>convg_thres) % && diff_max>tol_thres)
    niter=niter+1;
    
    % record last iter
%     loglik_old=loglik;
    V_joint_old=V_joint;
    grandV_old=grandV;
    V1_old=V_ind{1};
    V2_old=V_ind{2};

    
    % E step
    % some critical values
    grandse2=[];% 1*sum(p)
    grandSf=diag(Sf0)';% 1*(r0+sum(r))
%     grandV_temp=[];% combination of loadings: first few long columns are joint loadings, subsequent diagonal blocks are individual loadings
    grandB_temp=[];% concatenate B
    Delta_temp=ones(1,r0)*sum(se2.^(-1))/K;
    for k=1:K
        grandse2=[grandse2,(ones(1,p(k))*se2(k))];
        grandSf=[grandSf,diag(Sf{k})'];
%         grandV_temp=blkdiag(grandV_temp,V_ind{k});
        grandB_temp=[grandB_temp,B{k}];
        Delta_temp=[Delta_temp,ones(1,r(k))*(1/se2(k))];
    end;
    grandse2_inv=1./grandse2;% 1*sum(p)
    grandSf_inv=1./grandSf;% 1*(r0+sum(r))
%     grandV=[V_joint,grandV_temp];% sum(p)*(r0+sum(r))
    grandB=[B0,grandB_temp];%q*(r0+sum(r))
      Delta1=Delta_temp; % bsxfun(@times, grandV', grandse2_inv)*grandV; % 1*(r0+sum(r))
      Delta2=grandSf_inv+Delta1; % 1*(r0+sum(r))
      temp=bsxfun(@times,grandV,(1./Delta2))*grandV'; %sum(p)*sum(p)
    SigmaY_inv=diag(grandse2_inv)-bsxfun(@times,bsxfun(@times,temp,grandse2_inv),grandse2_inv'); %sum(p)*sum(p) ,not diagonal because of common structure, diff from SupSVD
    VSigmaYinvV=Delta1-Delta1.*(1./Delta2).*Delta1; % 1*(r0+sum(r))
    EU=X*bsxfun(@times,grandB, (ones(1,sum(r)+r0)-VSigmaYinvV.*grandSf))+grandY*SigmaY_inv*bsxfun(@times,grandV,grandSf); % conditional mean
    covU=grandSf-grandSf.*VSigmaYinvV.*grandSf; % 1*(r0+sum(r)), conditional variance (turns out to be a diagonal matrix)
  
    
    
    % M step
    % V and se2
    for k=1:K
        loc1=r0+sum(r(1:(k-1)))+1;
        loc2=loc1+r(k)-1;
        loc3=sum(p(1:(k-1)))+1;
        loc4=sum(p(1:k));
        Ycurrent=Y{k};
        EUcurrent=[EU(:,1:r0),EU(:,loc1:loc2)];
        EUcurrent_star=[(1/sqrt(K))*EU(:,1:r0),EU(:,loc1:loc2)];
        % V
          [tempL,~,tempR]=svds(Ycurrent'*EUcurrent_star,(r0+r(k)));
          Vcurrent_star=tempL*tempR'; % should have orthonormal columns
        V_joint(loc3:loc4,:)=Vcurrent_star(:,1:r0)*(1/sqrt(K));
        V_ind{k}=Vcurrent_star(:,(r0+1):(r0+r(k)));
        Vcurrent=[V_joint(loc3:loc4,:),V_ind{k}];
        clear EUcurrent_star Vcurrent_star;
        
        % se2 (directly from SupSVD formula)
          covUcurrent=covU([1:r0,loc1:loc2]); % 1*(r0+r(k))
          temp1=trace(Ycurrent*Ycurrent');
          temp2=2*trace(EUcurrent*Vcurrent'*Ycurrent');
          temp3=n*trace((Vcurrent'*Vcurrent)*diag(covUcurrent));
          temp4=trace((EUcurrent'*EUcurrent)*(Vcurrent'*Vcurrent));
        se2(k)=(temp1-temp2+temp3+temp4)/(n*p(k));
    end;
    
    % B and Sf
      EUcurrent=EU(:,1:r0);
    if sparsity
%         disp(['Start estimating B0...']);
        for i=1:r0
            [SpParam,FitInfo]=lasso(X,EUcurrent(:,i),'LambdaRatio',0,'Standardize',true); 
            BIC_score=n*log(FitInfo.MSE)+log(n)*FitInfo.DF;
            [~,ind]=min(BIC_score);
            B0(:,i)=SpParam(:,ind);
        end;
%         disp(['Finish estimating B0!']);
    else % if sparsity=0, no B-sparsity
        B0=(X'*X)\X'*EUcurrent;
    end;
      covUcurrent=covU(1:r0);
      temp1=n*diag(covUcurrent);
      temp2=EUcurrent'*EUcurrent;
      temp3=B0'*(X'*X)*B0;
      temp4=B0'*X'*EUcurrent;
      temp5=EUcurrent'*X*B0;
    Sf0=diag(diag(temp1+temp2+temp3-temp4-temp5)/n);
    for k=1:K
        loc1=r0+sum(r(1:(k-1)))+1;
        loc2=loc1+r(k)-1;
          EUcurrent=EU(:,loc1:loc2);
        if sparsity
%           disp(['Start estimating B',num2str(k),'...']);
          for i=1:r(k)
              [SpParam,FitInfo]=lasso(X,EUcurrent(:,i),'LambdaRatio',0,'Standardize',true); 
              BIC_score=n*log(FitInfo.MSE)+log(n)*FitInfo.DF;
              [~,ind]=min(BIC_score);
              B{k}(:,i)=SpParam(:,ind);
          end;
%           disp(['Finish estimating B',num2str(k),'!']);         
        else % if sparsity=0, no B-sparsity
          B{k}=(X'*X)\X'*EUcurrent;
        end;        
          covUcurrent=covU(loc1:loc2);
          temp1=n*diag(covUcurrent);
          temp2=EUcurrent'*EUcurrent;
          temp3=B{k}'*(X'*X)*B{k};
          temp4=B{k}'*X'*EUcurrent;
          temp5=EUcurrent'*X*B{k};
        Sf{k}=diag(diag(temp1+temp2+temp3-temp4-temp5)/n);
    end;
    
    % reorder columns of B, V, and rows/columns of Sf
    [temp,I]=sort(diag(Sf0),'descend');Sf0=diag(temp);
    B0=B0(:,I);V_joint=V_joint(:,I);
    for k=1:K
        [temp,I]=sort(diag(Sf{k}),'descend');Sf{k}=diag(temp);
        B{k}=B{k}(:,I);V_ind{k}=V_ind{k}(:,I);
    end;
    % calc grandV
    grandV=[];
    for k=1:K
        grandV=blkdiag(grandV,V_ind{k});
    end;
    grandV=[V_joint,grandV];% sum(p)*(r0+sum(r))

    
    % stopping rule
    diff=PrinAngle(grandV,grandV_old);
    recdiff=[recdiff,diff];
%     diff
    
    % draw loglik trend
    loglik=loglikelihood(X,Y,B0,B,V_joint, V_ind,se2,Sf0,Sf);
    diff_max=loglik-maxloglik; % insurance, avoid likelihood decrease
    maxloglik=max(maxloglik,loglik);
    
%     % check
%     recloglik=[recloglik,loglik];
%     figure(100);
%     subplot(2,1,1)
%     plot(recloglik,'o-');title('likelihood');
%     subplot(2,1,2)
%     plot(recdiff,'o-');title('angle');
%     drawnow

    % check
%     
%     recdiff_grandV=[recdiff_grandV,180/pi*acos(min(svd(grandV_old'*grandV)))];
%     recdiff_V_joint=[recdiff_V_joint,PrinAngle(V_joint,V_joint_old)];
%     recdiff_V1=[recdiff_V1,PrinAngle(V_ind{1},V1_old)];
%     recdiff_V2=[recdiff_V2,PrinAngle(V_ind{2},V2_old)];
%     figure(101);
%     subplot(2,2,1)
%     plot(recdiff_grandV,'o-');
%     subplot(2,2,2)
%     plot(recdiff_V_joint,'+-');
%     subplot(2,2,3)
%     plot(recdiff_V1,'+-');
%     subplot(2,2,4)
%     plot(recdiff_V2,'+-');
%     drawnow
    
end;



% calc EU
grandse2=[];% 1*sum(p)
grandSf=diag(Sf0)';% 1*(r0+sum(r))
grandV_temp=[];% combination of loadings: first few long columns are joint loadings, subsequent diagonal blocks are individual loadings
grandB_temp=[];% concatenate B
Delta_temp=ones(1,r0)*sum(se2.^(-1))/K;
for k=1:K
    grandse2=[grandse2,(ones(1,p(k))*se2(k))];
    grandSf=[grandSf,diag(Sf{k})'];
    grandV_temp=blkdiag(grandV_temp,V_ind{k});
    grandB_temp=[grandB_temp,B{k}];
    Delta_temp=[Delta_temp,ones(1,r(k))*(1/se2(k))];
end;
grandse2_inv=1./grandse2;% 1*sum(p)
grandSf_inv=1./grandSf;% 1*(r0+sum(r))
grandV=[V_joint,grandV_temp];% sum(p)*(r0+sum(r))
grandB=[B0,grandB_temp];%q*(r0+sum(r))
  Delta1=Delta_temp; % bsxfun(@times, grandV', grandse2_inv)*grandV; % 1*(r0+sum(r))
  Delta2=grandSf_inv+Delta1; % 1*(r0+sum(r))
  temp=bsxfun(@times,grandV,(1./Delta2))*grandV'; %sum(p)*sum(p)
SigmaY_inv=diag(grandse2_inv)-bsxfun(@times,bsxfun(@times,temp,grandse2_inv),grandse2_inv'); %sum(p)*sum(p) ,not diagonal because of common structure, diff from SupSVD
VSigmaYinvV=Delta1-Delta1.*(1./Delta2).*Delta1; % 1*(r0+sum(r))
EU=X*bsxfun(@times,grandB, (ones(1,sum(r)+r0)-VSigmaYinvV.*grandSf))+grandY*SigmaY_inv*bsxfun(@times,grandV,grandSf); % conditional mean

    
    
    
% Print convergence information
if niter<max_niter
    disp(['SIPCA_B converges after ',num2str(niter),' iterations.']);
else
    disp(['SIPCA_B NOT converge after ',num2str(max_niter),' iterations!!! Final change in angle: ',num2str(diff)]);
end;

end







function [Q1,Q,R] = GramSchmidt(A1,B)
% A1 = randn(p1+p2,r0);
% compute QR of A1 using Gram-Schmidt, orthogonal to (orthogonal) B.

[m,n] = size(A1);

if nargin == 2
    [m1,n2] = size(B);
    
    if m ~=m1; return;
    end
    
    A=[B A1];
    
    for j = 1:n+n2
        v = A(:,j);
        for i=1:j-1
            R(i,j) = Q(:,i)'*A(:,j);
            v = v - R(i,j)*Q(:,i);
        end
        R(j,j) = norm(v);
        Q(:,j) = v/R(j,j);
    end
    
    Q1= Q(:,n2+1:end);
    
    
else
    A = A1;
    
    for j = 1:n
        v = A(:,j);
        for i=1:j-1
            R(i,j) = Q(:,i)'*A(:,j);
            v = v - R(i,j)*Q(:,i);
        end
        R(j,j) = norm(v);
        Q(:,j) = v/R(j,j);
    end
    
    Q1= Q;
    
end
end





function angle = PrinAngle(V1,V2,paramstruct)
% if ind=1 (default), calc max principal angle
% if ind=2, Calculates All principal angles between column space of V1 and V2
ind=1;
if nargin > 2 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'ind');
      ind=getfield(paramstruct,'ind');
  end;
end;

[p1,r1]=size(V1);
[p2,r2]=size(V2);
if (p1~=p2) 
    error('Input must be matched')
end;

[V1,~,~]=svd(V1,'econ');
[V2,~,~]=svd(V2,'econ');
if ind==1
    angle=180/pi*acos(min(svd(V1'*V2)));
elseif ind==2
    angle=180/pi*acos(svd(V1'*V2));
end;

end