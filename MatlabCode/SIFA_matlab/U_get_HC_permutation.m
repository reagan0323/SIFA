function z = U_get_HC_permutation(Lgene)

m = size(Lgene,1) + 1;

Gcount = [ones(m,1); zeros(m-1,1)];

for g=1:(m-1)
	Gcount(g+m)  = Gcount(Lgene(g,1))  + Gcount(Lgene(g,2));
end;

StartingPos = -ones(size(Gcount));
StartingPos(end) = 1;


for i=(m-1):-1:1
	LeftLength  = Gcount(Lgene(i,1));
	% RightLength = Gcount(Lgene(i,2)); 
	% total length = Gcount(i+m)
	
	StartingPos(Lgene(i,1)) = StartingPos(i+m);
	StartingPos(Lgene(i,2)) = StartingPos(i+m) + LeftLength;
end

PermInv = StartingPos(1:m);

[a Perm] = sort(PermInv);

z = Perm;

% error check: all(sort(z)' == 1:m)