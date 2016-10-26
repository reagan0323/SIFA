function loss = GrassDist(V1,V2)
% Calculates the grassmannian distance between column space of V1 and V2

[p1,r1]=size(V1);
[p2,r2]=size(V2);
if (p1~=p2) 
    error('Input must be matched')
end;

[V1,~,~]=svd(V1,'econ'); % do NOT use svds for min(n,p) decomp of n*p matrix, use GramSchmidt or svd('econ') instead
[V2,~,~]=svd(V2,'econ');

% loss=norm(acos(svd(V1'*V2)),'fro');

% Irina 03/02/16: I think this should be adjusted as below to compare
% different subspaces
loss=norm(acos(svd(V1'*V2,'econ')),2);

end

   