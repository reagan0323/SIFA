function [V_joint, V_ind,se2,Sf0,Sf,EU,fX]=SICA_A_np(X,Y,r0,r,paramstruct)
% This function fits SIPCA under the general conditions.
% 1. V0k full column rank
% 2. V0k and Vk have independent columns
% It assumes a user-specified relation between U and X
% This subsumes SIPCA_A as a special case (with relation='linear')
% (be caureful about slightly different output)
%
% Input: 
%       X           n*q matrix, centered covariate data 
%       Y           1*K cell array, each cell is a n*pi centered primary
%                   data, should roughly contribute equally to joint struct
%       r0          scalar, prespecified rank of common structure
%       r           1*K vector, prespecified rank of specific structures
%       paramstruct
%            relation    'linear' (default), use linear function to model
%                           the relation between U and covariates
%                        'univ_kernel', use kernel methods for *single* covariates
%                        more upcoming, can be defined by users
%
%            sparsity    1 , when est B0 or B, use LASSO with BIC
%                           to select the best tuning, suitable for high dimension
%                        0 (default), no sparsity, only valid for low dimension
%                        only valid when relation='linear'
%                     
%            Niter       default 1000, max number of iteration
%            Tol         default 0.01, threshold for grandV max PrinAngle change
%
%
% Output:
%       V_joint     sum(p)*r0 matrix, stacked joint loadings, with orthonormal
%                   columns. 
%       V_ind       1*K array, each is a pi*ri loading matrix, with
%                   orthonormal columns
%       se2         1*K vector, noise variance for each phenotypic data set
%       Sf0         r0*r0 matrix, diagonal covariance matrix
%       Sf          1*K array, each is a ri*ri diagonal covariance matrix
%       EU          n*(r0+sum(ri)) matrix, conditional expectation of joint and individual scores
%       fX          optional output, n*(r0+sum(ri)) matrix, the
%                   deterministic part of scores
%
% Created: 2016.3.19
% By: Gen Li
% 



relation='linear';
sparsity=0;
max_niter=1000;
convg_thres=0.01; 
if nargin > 4 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'relation');
      relation=getfield(paramstruct,'relation');
  end;
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




% initial value
grandY=[];
for k=1:K
    grandY=[grandY,Y{k}];
end;
grandU=[];
grandV=[];
[U_joint_ini,D_joint_ini,V_joint]=svds(grandY,r0); % note: segments of V_joint_ini corresponding to each data set may not be orthogonal as desired
U_joint=U_joint_ini*D_joint_ini;
V_ind={};
Sf={};
for k=1:K
    loc3=sum(p(1:(k-1)))+1;
    loc4=sum(p(1:k));
    Vcurrent=V_joint(loc3:loc4,:);
    Ycurrent=Y{k}-U_joint*Vcurrent';
    [U_k_ini,D_k_ini,V_k]=svds(Ycurrent,r(k));
    U_k=U_k_ini*D_k_ini;
    V_ind{k}=V_k;
    grandU=[grandU,U_k];
    grandV=blkdiag(grandV,V_ind{k});
end;
grandU=[U_joint,grandU];
grandV=[V_joint,grandV];% sum(p)*(r0+sum(r))
fX=EstU_fun(X,grandU,relation,sparsity); % estimate an n*(r0+sum(ri)) matrix
for k=1:K
    loc1=r0+sum(r(1:(k-1)))+1;
    loc2=loc1+r(k)-1;
    loc3=sum(p(1:(k-1)))+1;
    loc4=sum(p(1:k));
    U_k=grandU(:,loc1:loc2);
    fkX=fX(:,loc1:loc2);
    Sf{k}=diag(std(U_k-fkX).^2);
    temp=grandU*grandV';
    se2(k)=norm(Y{k}-temp(:,loc3:loc4),'fro')^2/(n*p(k));
end;
Sf0=diag(std(U_joint-fX(:,1:r0)).^2);



% disp('Initial est done!')
loglik=loglikelihood1(Y,fX,V_joint, V_ind,se2,Sf0,Sf);
recloglik=loglik;
maxloglik=loglik;



niter=0; 
diff=inf; 
% diff_max=inf; 
% tol_thres=-10;
recdiff=[];
while (niter<=max_niter && abs(diff)>convg_thres) %&& diff_max>tol_thres)
    niter=niter+1;
    
    % record last iter
    grandV_old=grandV;



    % E step
    % some critical values
    grandse2=[];% 1*sum(p)
    grandSf0=diag(Sf0)';% 1*r0
    grandSf=[];% 1*sum(r)
    Delta_temp=[]; % 1*sum(r)
    for k=1:K
        grandse2=[grandse2,(ones(1,p(k))*se2(k))];
        grandSf=[grandSf,diag(Sf{k})'];
        Delta_temp=[Delta_temp,ones(1,r(k))*(1/se2(k))];
    end;
    grandse2_inv=1./grandse2;% 1*sum(p)
    grandSf_inv=1./grandSf;% 1*sum(r)
    grandSf0_inv=1./grandSf0; % 1*r0
      Delta1=bsxfun(@times,grandV',grandse2_inv)*grandV; % [r0+sum(r)]*[r0+sum(r)]
      Delta2_inv=inv(diag([grandSf0_inv,grandSf_inv])+Delta1);
      temp=grandV*Delta2_inv*grandV'; %sum(p)*sum(p)
    SigmaY_inv=diag(grandse2_inv)-bsxfun(@times,bsxfun(@times,temp,grandse2_inv),grandse2_inv'); %sum(p)*sum(p) ,not diagonal because of common structure, diff from SupSVD
    VSigmaYinvV=Delta1-Delta1*Delta2_inv*Delta1; % (r0+sum(r))*(r0+sum(r))
    EU=fX*(eye(sum(r)+r0)-bsxfun(@times,VSigmaYinvV, [grandSf0,grandSf]))+grandY*SigmaY_inv*bsxfun(@times,grandV,[grandSf0,grandSf]); % n*(r0+sum(r)), conditional mean
    covU=diag([grandSf0,grandSf])-bsxfun(@times,bsxfun(@times,VSigmaYinvV, [grandSf0,grandSf]),[grandSf0,grandSf]'); % (r0+sum(r))*(r0+sum(r))
  
    
    
    % M step
    % est V
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
    
    
    
    
    
    
    % est fX 
    fX=EstU_fun(X,EU,relation,sparsity); % estimate an n*(r0+sum(ri)) matrix


    % est Sf0 and Sf
    f0X=fX(:,1:r0);
    EUcurrent=EU(:,1:r0);
    covUcurrent=covU(1:r0,1:r0);
    temp1=n*covUcurrent;
    temp2=EUcurrent'*EUcurrent;
    temp3=f0X'*f0X;
    temp4=f0X'*EUcurrent;
    temp5=EUcurrent'*f0X;
%     Sf0=(temp1+temp2+temp3-temp4-temp5)/n;                   % questionable!!!
    Sf0=diag(diag(temp1+temp2+temp3-temp4-temp5)/n); % exactly follow the draft
    for k=1:K
        loc1=r0+sum(r(1:(k-1)))+1;
        loc2=loc1+r(k)-1;
        fkX=fX(:,loc1:loc2);
        EUcurrent=EU(:,loc1:loc2);            
        covUcurrent=covU(loc1:loc2,loc1:loc2);
        temp1=n*covUcurrent;
        temp2=EUcurrent'*EUcurrent;
        temp3=fkX'*fkX;
        temp4=fkX'*EUcurrent;
        temp5=EUcurrent'*fkX;
        Sf{k}=diag(diag((temp1+temp2+temp3-temp4-temp5)/n));
    end;
    
          
    % Post Standardization for V0 (and Sf0 and B0)
    [V_joint_new,Sf0,~]=svds(V_joint*Sf0*V_joint',r0);
    fX(:,1:r0)=fX(:,1:r0)*V_joint'*V_joint_new;              % NEED TO MENTION THIS IN DRAFT
    V_joint=V_joint_new;
    % reorder columns of B, V, and rows/columns of Sf
    for k=1:K
        [temp,I]=sort(diag(Sf{k}),'descend');Sf{k}=diag(temp);
        loc1=r0+sum(r(1:(k-1)))+1;
        loc2=loc1+r(k)-1;
        temp=fX(:,loc1:loc2);
        fX(:,loc1:loc2)=temp(:,I);
        V_ind{k}=V_ind{k}(:,I);
    end;
    % get grandV
    grandV=[];
    for k=1:K
        grandV=blkdiag(grandV,V_ind{k});
    end;
    grandV=[V_joint,grandV];% sum(p)*(r0+sum(r))
    
        
    
    % stopping rule
    diff=PrinAngle(grandV,grandV_old);
    recdiff=[recdiff,diff];
    
    % draw loglik trend
    loglik=loglikelihood1(Y,fX,V_joint, V_ind,se2,Sf0,Sf);
    diff_max=loglik-maxloglik; % insurance, avoid likelihood decrease
    maxloglik=max(maxloglik,loglik);
    
    % check
    recloglik=[recloglik,loglik];
    figure(100);
    subplot(2,1,1)
    plot(recloglik,'o-');title('likelihood');
    subplot(2,1,2)
    plot(recdiff,'o-');title(['angle=',num2str(diff)]);
    drawnow

end;

% calculate EU
grandse2=[];% 1*sum(p)
grandSf0=diag(Sf0)';% 1*r0
grandSf=[];% 1*sum(r)
grandV_temp=[];% combination of loadings: first few long columns are joint loadings, subsequent diagonal blocks are individual loadings
Delta_temp=[]; % 1*sum(r)
for k=1:K
    grandse2=[grandse2,(ones(1,p(k))*se2(k))];
    grandSf=[grandSf,diag(Sf{k})'];
    grandV_temp=blkdiag(grandV_temp,V_ind{k});
    Delta_temp=[Delta_temp,ones(1,r(k))*(1/se2(k))];
end;
grandse2_inv=1./grandse2;% 1*sum(p)
grandSf_inv=1./grandSf;% 1*sum(r)
grandSf0_inv=1./grandSf0; % 1*r0
grandV=[V_joint,grandV_temp];% sum(p)*(r0+sum(r))
  Delta1=bsxfun(@times,grandV',grandse2_inv)*grandV; % [r0+sum(r)]*[r0+sum(r)]
  Delta2_inv=inv(diag([grandSf0_inv,grandSf_inv])+Delta1);
  temp=grandV*Delta2_inv*grandV'; %sum(p)*sum(p)
SigmaY_inv=diag(grandse2_inv)-bsxfun(@times,bsxfun(@times,temp,grandse2_inv),grandse2_inv'); %sum(p)*sum(p) ,not diagonal because of common structure, diff from SupSVD
VSigmaYinvV=Delta1-Delta1*Delta2_inv*Delta1; % (r0+sum(r))*(r0+sum(r))
EU=fX*(eye(sum(r)+r0)-bsxfun(@times,VSigmaYinvV, [grandSf0,grandSf]))+grandY*SigmaY_inv*bsxfun(@times,grandV,[grandSf0,grandSf]); % n*(r0+sum(r)), conditional mean


% Print convergence information
if niter<max_niter
    disp(['SIPCA_A1 converges after ',num2str(niter),' iterations.']);
else
    disp(['SIPCA_A1 NOT converge after ',num2str(max_niter),' iterations!!! Final change in angle: ',num2str(diff)]);
end;

end



function out=loglikelihood1(Y,fX,V_joint, V_ind,se2,Sf0,Sf)
% This function calc sum of log likelihood of Y 
% The output is proportional to the sum of log likelihood (without the constant term)
% The larger the better!!!!!


% get dimension
[n,~]=size(fX);
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
for k=1:K
    grandse2=[grandse2,(ones(1,p(k))*se2(k))];
    grandSf=[grandSf,diag(Sf{k})'];
    grandV_temp=blkdiag(grandV_temp,V_ind{k});
end;
grandse2_inv=1./grandse2;% 1*sum(p)
grandSf_inv=1./grandSf;% 1*sum(r)
grandSf0_inv=1./grandSf0; % 1*r0
grandV=[V_joint,grandV_temp];% sum(p)*(r0+sum(r))
  Delta1=bsxfun(@times,grandV',grandse2_inv)*grandV; % [r0+sum(r)]*[r0+sum(r)]
  Delta2_inv=inv(diag([grandSf0_inv,grandSf_inv])+Delta1);
  temp=grandV*Delta2_inv*grandV'; %sum(p)*sum(p)
SigmaY_inv=diag(grandse2_inv)-bsxfun(@times,bsxfun(@times,temp,grandse2_inv),grandse2_inv'); %sum(p)*sum(p) ,not diagonal because of common structure, diff from SupSVD
temp=bsxfun(@times,grandV,sqrt([grandSf0,grandSf]));
SigmaY=diag(grandse2)+temp*temp';

% loglikelihood terms
term1=-n*sum(log(diag(chol(SigmaY)))); % log det
temp=grandY-fX*grandV';
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



function fX=EstU_fun(X,U,relation,sparsity)
[~,q]=size(X);
[n,r]=size(U);
fX=zeros(size(U));
if strcmpi(relation,'linear') && sparsity==0 % linear non-sparse
    B=(X'*X)\X'*U; % q*sum(r)
    fX=X*B; % naturally column centered  b/c X are centered
    
elseif strcmpi(relation,'linear') && sparsity==1 % linear and sparse
    B=zeros(q,r);
    for i=1:r % for each column of U
        [SpParam,FitInfo]=lasso(X,U(:,i),'LambdaRatio',0,'Standardize',true); 
        BIC_score=n*log(FitInfo.MSE)+log(n)*FitInfo.DF;
        [~,ind]=min(BIC_score);
        B(:,i)=SpParam(:,ind);
    end;
    fX=X*B;% naturally column centered  b/c X are centered
    
elseif strcmpi(relation,'univ_kernel')
    if q~=1
        error('Cannot deal with multivariate covariates...')
    end;
    for i=1:r % for each column of U
       out=ksr(X,U(:,i));
       fX(:,i)=out.f;
    end;
    % center each column of fX (b/c we assume X and Y_i are all column centered)
    fX=bsxfun(@minus,fX,mean(fX,1));
else
    error('No such relation function available...')
end;

end
