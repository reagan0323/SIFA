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

[V1,~,~]=svd(V1,'econ'); % do NOT use svds for min(n,p) decomp of n*p matrix, use GramSchmidt or svd('econ') instead
[V2,~,~]=svd(V2,'econ');
if ind==1
    angle=180/pi*acos(min(svd(V1'*V2,'econ')));
elseif ind==2
    angle=180/pi*acos(svd(V1'*V2,'econ'));
end;

end
