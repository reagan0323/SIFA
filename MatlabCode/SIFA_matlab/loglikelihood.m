
function out=loglikelihood(X,Y,B0,B,V_joint, V_ind,se2,Sf0,Sf)
%
% This function calc sum of log likelihood of Y 
% The input is data and the output of SIPCA_XXX
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
