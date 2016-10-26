function [B0,B,V_joint, V_ind,se2,Sf0,Sf,EU]=SIFA_A(X,Y,r0,r,paramstruct)
% This function fits SIPCA under the general conditions.
% 1. V0k full column rank
% 2. V0k and Vk have independent columns
% It assumes a linear relation, and may or may not impose sparsity on B
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
%            Tol         default 1E-3,
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
% 

sparsity=1;
max_niter=500;
convg_thres=1E-3; 
flag_true=0;
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
  % courtesy check
  if isfield(paramstruct,'trueV0') ;    
    trueV0 = getfield(paramstruct,'trueV0') ;
    trueV1 = getfield(paramstruct,'trueV1') ;
    trueV2 = getfield(paramstruct,'trueV2') ;
    flag_true=1;
    inner1=[];
    inner2=[];
    inner3=[];
  end ;
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




% initial value
grandY=[];
for k=1:K
    grandY=[grandY,Y{k}];
end;
[U_joint_ini,D_joint_ini,V_joint]=svds(grandY,r0); % note: segments of V_joint_ini corresponding to each data set may not be orthogonal as desired
U_joint=U_joint_ini*D_joint_ini;
if sparsity
    for i=1:r0
        [SpParam,FitInfo]=lasso(X,U_joint(:,i),'LambdaRatio',0,'Standardize',true); BIC_score=n*log(FitInfo.MSE)+log(n)*FitInfo.DF;
        [~,ind]=min(BIC_score);
        B0(:,i)=SpParam(:,ind);
    end;
else % if sparsity=0, no B-sparsity
    B0=(X'*X)\X'*U_joint;
end;
Sf0=diag(std(U_joint-X*B0).^2);
V_ind={};
B={};
Sf={};
for k=1:K
    loc3=sum(p(1:(k-1)))+1;
    loc4=sum(p(1:k));
    Vcurrent=V_joint(loc3:loc4,:);
    Ycurrent=Y{k}-U_joint*Vcurrent';
    [U_k_ini,D_k_ini,V_k]=svds(Ycurrent,r(k));
    U_k=U_k_ini*D_k_ini;
    V_ind{k}=V_k;
    if sparsity
        for i=1:r(k)
            [SpParam,FitInfo]=lasso(X,U_k(:,i),'LambdaRatio',0,'Standardize',true); 
            BIC_score=n*log(FitInfo.MSE)+log(n)*FitInfo.DF;
            [~,ind]=min(BIC_score);
            B{k}(:,i)=SpParam(:,ind);
        end;
    else % if sparsity=0, no B-sparsity
        B{k}=(X'*X)\X'*U_k;
    end;
    Sf{k}=diag(std(U_k-X*B{k}).^2);
    se2(k)=norm(Y{k}-U_k*V_k'-U_joint*Vcurrent','fro')^2/(n*p(k));
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
while (niter<=max_niter && abs(diff)>convg_thres) %&& diff_max>tol_thres)
    niter=niter+1;
    
    % record last iter
%     loglik_old=loglik;
%     V_joint_old=V_joint;
    grandV_old=grandV;



    % E step
    % some critical values
    grandse2=[];% 1*sum(p)
    grandSf0=diag(Sf0)';% 1*r0
    grandSf=[];% 1*sum(r)
%     grandV_temp=[];% combination of loadings: first few long columns are joint loadings, subsequent diagonal blocks are individual loadings
    grandB_temp=[];% concatenate B
    Delta_temp=[]; % 1*sum(r)
    for k=1:K
        grandse2=[grandse2,(ones(1,p(k))*se2(k))];
        grandSf=[grandSf,diag(Sf{k})'];
%         grandV_temp=blkdiag(grandV_temp,V_ind{k});
        grandB_temp=[grandB_temp,B{k}];
        Delta_temp=[Delta_temp,ones(1,r(k))*(1/se2(k))];
    end;
    grandse2_inv=1./grandse2;% 1*sum(p)
    grandSf_inv=1./grandSf;% 1*sum(r)
    grandSf0_inv=1./grandSf0; % 1*r0
%     grandV=[V_joint,grandV_temp];% sum(p)*(r0+sum(r))
    grandB=[B0,grandB_temp];%q*(r0+sum(r))
      Delta1=bsxfun(@times,grandV',grandse2_inv)*grandV; % [r0+sum(r)]*[r0+sum(r)]
      Delta2_inv=inv(diag([grandSf0_inv,grandSf_inv])+Delta1);
      temp=grandV*Delta2_inv*grandV'; %sum(p)*sum(p)
    SigmaY_inv=diag(grandse2_inv)-bsxfun(@times,bsxfun(@times,temp,grandse2_inv),grandse2_inv'); %sum(p)*sum(p) ,not diagonal because of common structure, diff from SupSVD
    VSigmaYinvV=Delta1-Delta1*Delta2_inv*Delta1; % (r0+sum(r))*(r0+sum(r))
    EU=X*grandB*(eye(sum(r)+r0)-bsxfun(@times,VSigmaYinvV, [grandSf0,grandSf]))+grandY*SigmaY_inv*bsxfun(@times,grandV,[grandSf0,grandSf]); % n*(r0+sum(r)), conditional mean
    covU=diag([grandSf0,grandSf])-bsxfun(@times,bsxfun(@times,VSigmaYinvV, [grandSf0,grandSf]),[grandSf0,grandSf]'); % (r0+sum(r))*(r0+sum(r))
  
    
    
    % M step
    % V and se2
%   for iter=1:3 % alternate between V_joint and V_ind
    for k=1:K
        loc1=r0+sum(r(1:(k-1)))+1;
        loc2=loc1+r(k)-1;
        loc3=sum(p(1:(k-1)))+1;
        loc4=sum(p(1:k));
 
        % critical value
        EU0=EU(:,1:r0);
        EUk=EU(:,loc1:loc2);
        EUkU0=n*covU(loc1:loc2,1:r0)+EUk'*EU0; % r(k)*r0
        EU0U0=n*covU(1:r0,1:r0)+EU0'*EU0; % r0*r0
        Vk=V_ind{k}; % p(k)*r(k)
        
        % V_0k
        V_0k=(Y{k}'*EU0-Vk*EUkU0)/EU0U0;
        V_joint(loc3:loc4,:)=V_0k;
        
        % V_k
        [tempL,~,tempR]=svds(EUk'*Y{k}-EUkU0*V_0k',r(k));
        V_ind{k}=tempR*tempL';
    end;
%   end;

    
    % est se2
    for k=1:K
        loc1=r0+sum(r(1:(k-1)))+1;
        loc2=loc1+r(k)-1;
        loc3=sum(p(1:(k-1)))+1;
        loc4=sum(p(1:k));
        covUcurrent=covU([1:r0,loc1:loc2],[1:r0,loc1:loc2]); % (r0+r(k))*(r0+r(k))
        Ycurrent=Y{k};
        EUcurrent=[EU(:,1:r0),EU(:,loc1:loc2)];
        Vcurrent=[V_joint(loc3:loc4,:),V_ind{k}];
        temp1=trace(Ycurrent*Ycurrent');
        temp2=2*trace(EUcurrent*Vcurrent'*Ycurrent');
        temp3=n*trace((Vcurrent'*Vcurrent)*covUcurrent);
        temp4=trace((EUcurrent'*EUcurrent)*(Vcurrent'*Vcurrent));
        se2(k)=(temp1-temp2+temp3+temp4)/(n*p(k));
    end;
    
    
    
    
    
    
    % B and Sf
      EUcurrent=EU(:,1:r0);
    if sparsity
        for i=1:r0
            [SpParam,FitInfo]=lasso(X,EUcurrent(:,i),'LambdaRatio',0,'Standardize',true); 
            BIC_score=n*log(FitInfo.MSE)+log(n)*FitInfo.DF;
            [~,ind]=min(BIC_score);
            B0(:,i)=SpParam(:,ind);
        end;
    else % if sparsity=0, no B-sparsity
        B0=(X'*X)\X'*EUcurrent;
    end;
      covUcurrent=covU(1:r0,1:r0);
      temp1=n*covUcurrent;
      temp2=EUcurrent'*EUcurrent;
      temp3=B0'*(X'*X)*B0;
      temp4=B0'*X'*EUcurrent;
      temp5=EUcurrent'*X*B0;
    Sf0=(temp1+temp2+temp3-temp4-temp5)/n; % not diagonal
    for k=1:K
        loc1=r0+sum(r(1:(k-1)))+1;
        loc2=loc1+r(k)-1;
          EUcurrent=EU(:,loc1:loc2);
        if sparsity
          for i=1:r(k)
              [SpParam,FitInfo]=lasso(X,EUcurrent(:,i),'LambdaRatio',0,'Standardize',true); 
              BIC_score=n*log(FitInfo.MSE)+log(n)*FitInfo.DF;
              [~,ind]=min(BIC_score);
              B{k}(:,i)=SpParam(:,ind);
          end;
        else % if sparsity=0, no B-sparsity
          B{k}=(X'*X)\X'*EUcurrent;
        end;                
          covUcurrent=covU(loc1:loc2,loc1:loc2);
          temp1=n*covUcurrent;
          temp2=EUcurrent'*EUcurrent;
          temp3=B{k}'*(X'*X)*B{k};
          temp4=B{k}'*X'*EUcurrent;
          temp5=EUcurrent'*X*B{k};
        Sf{k}=diag(diag((temp1+temp2+temp3-temp4-temp5)/n));
    end;
    
          
    % Post Standardization for V0 (and Sf0 and B0)
    [V_joint_new,Sf0,~]=svds(V_joint*Sf0*V_joint',r0);
    V_joint_new=bsxfun(@times,V_joint_new,sign(V_joint_new(1,:)));% make sure no sign alternation
    B0=B0*V_joint'*V_joint_new; % will sacrifice any sparsity in B0
    V_joint=V_joint_new;
    % reorder columns of B, V, and rows/columns of Sf
    for k=1:K
        [temp,I]=sort(diag(Sf{k}),'descend');Sf{k}=diag(temp);
        B{k}=B{k}(:,I);V_ind{k}=V_ind{k}(:,I);
    end;
    % get grandV
    grandV=[];
    for k=1:K
        grandV=blkdiag(grandV,V_ind{k});
    end;
    grandV=[V_joint,grandV];% sum(p)*(r0+sum(r))
    
    
        if flag_true==1
        figure(42);
        subplot(2,2,1)
        inner1=[inner1,180/pi*acos(min(svd(V_joint'*trueV0)))];
        plot(inner1');
        title('Principal Angle for Joint');
        subplot(2,2,3)
        inner2=[inner2,180/pi*acos(min(svd(V_ind{1}'*trueV1)))];
        plot(inner2');
        title('Principal Angle for Indiv 1')
        subplot(2,2,4)
        inner3=[inner3,180/pi*acos(min(svd(V_ind{2}'*trueV2)))];
        plot(inner3');
        title('Principal Angle for Indiv 2')
        end;
    
    
    % stopping rule
    diff=PrinAngle(grandV,grandV_old);
    recdiff=[recdiff,diff];
    
    % draw loglik trend
    loglik=loglikelihood(X,Y,B0,B,V_joint, V_ind,se2,Sf0,Sf);
    diff_max=loglik-maxloglik; % insurance, avoid likelihood decrease
    maxloglik=max(maxloglik,loglik);
    
%    % check
%    recloglik=[recloglik,loglik];
%     figure(100);
%     subplot(2,1,1)
%     plot(recloglik,'o-');title('likelihood');
%     subplot(2,1,2)
%     plot(recdiff,'o-');title('angle');
%     drawnow

end;

% calculate EU
grandse2=[];% 1*sum(p)
grandSf0=diag(Sf0)';% 1*r0
grandSf=[];% 1*sum(r)
grandV_temp=[];% combination of loadings: first few long columns are joint loadings, subsequent diagonal blocks are individual loadings
grandB_temp=[];% concatenate B
Delta_temp=[]; % 1*sum(r)
for k=1:K
    grandse2=[grandse2,(ones(1,p(k))*se2(k))];
    grandSf=[grandSf,diag(Sf{k})'];
    grandV_temp=blkdiag(grandV_temp,V_ind{k});
    grandB_temp=[grandB_temp,B{k}];
    Delta_temp=[Delta_temp,ones(1,r(k))*(1/se2(k))];
end;
grandse2_inv=1./grandse2;% 1*sum(p)
grandSf_inv=1./grandSf;% 1*sum(r)
grandSf0_inv=1./grandSf0; % 1*r0
grandV=[V_joint,grandV_temp];% sum(p)*(r0+sum(r))
grandB=[B0,grandB_temp];%q*(r0+sum(r))
  Delta1=bsxfun(@times,grandV',grandse2_inv)*grandV; % [r0+sum(r)]*[r0+sum(r)]
  Delta2_inv=inv(diag([grandSf0_inv,grandSf_inv])+Delta1);
  temp=grandV*Delta2_inv*grandV'; %sum(p)*sum(p)
SigmaY_inv=diag(grandse2_inv)-bsxfun(@times,bsxfun(@times,temp,grandse2_inv),grandse2_inv'); %sum(p)*sum(p) ,not diagonal because of common structure, diff from SupSVD
VSigmaYinvV=Delta1-Delta1*Delta2_inv*Delta1; % (r0+sum(r))*(r0+sum(r))
EU=X*grandB*(eye(sum(r)+r0)-bsxfun(@times,VSigmaYinvV, [grandSf0,grandSf]))+grandY*SigmaY_inv*bsxfun(@times,grandV,[grandSf0,grandSf]); % n*(r0+sum(r)), conditional mean


% Print convergence information
if niter<max_niter
    disp(['SIPCA_A converges after ',num2str(niter),' iterations.']);
else
    disp(['SIPCA_A NOT converge after ',num2str(max_niter),' iterations!!! Final change in angle: ',num2str(diff)]);
end;

end



function out=loglikelihood(X,Y,B0,B,V_joint, V_ind,se2,Sf0,Sf)
%
% This function calc sum of log likelihood of Y 
% The input is data and the output of SIPCA_LG_v
% The output is proportional to the sum of log likelihood (without the constant term)
%
% The larger the better!!!!!
%
% Created: 2015.10.4
% By: Gen Li

% get dimension
[n,q]=size(X);
K=length(Y);
p=zeros(1,K); 
for k=1:K
    [n_temp,p(k)]=size(Y{k});
    if n_temp~=n
        error('Sample mismatch!');
    end;
end;


% critical value
grandY=[];
for k=1:K
    grandY=[grandY,Y{k}];
end;
grandse2=[];% 1*sum(p)
grandSf0=diag(Sf0)';% 1*r0
grandSf=[];% 1*sum(r)
grandV_temp=[];% combination of loadings: first few long columns are joint loadings, subsequent diagonal blocks are individual loadings
grandB_temp=[];% concatenate B
for k=1:K
    grandse2=[grandse2,(ones(1,p(k))*se2(k))];
    grandSf=[grandSf,diag(Sf{k})'];
    grandV_temp=blkdiag(grandV_temp,V_ind{k});
    grandB_temp=[grandB_temp,B{k}];
end;
grandse2_inv=1./grandse2;% 1*sum(p)
grandSf_inv=1./grandSf;% 1*sum(r)
grandSf0_inv=1./grandSf0; % 1*r0
grandV=[V_joint,grandV_temp];% sum(p)*(r0+sum(r))
grandB=[B0,grandB_temp];%q*(r0+sum(r))
  Delta1=bsxfun(@times,grandV',grandse2_inv)*grandV; % [r0+sum(r)]*[r0+sum(r)]
  Delta2_inv=inv(diag([grandSf0_inv,grandSf_inv])+Delta1);
  temp=grandV*Delta2_inv*grandV'; %sum(p)*sum(p)
SigmaY_inv=diag(grandse2_inv)-bsxfun(@times,bsxfun(@times,temp,grandse2_inv),grandse2_inv'); %sum(p)*sum(p) ,not diagonal because of common structure, diff from SupSVD
temp=bsxfun(@times,grandV,sqrt([grandSf0,grandSf]));
SigmaY=diag(grandse2)+temp*temp';

% loglikelihood terms
term1=-n*sum(log(diag(chol(SigmaY)))); % log det
temp=grandY-X*grandB*grandV';
term2=-1/2*trace(SigmaY_inv*(temp'*temp));

% final output
out=term1+term2;
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
