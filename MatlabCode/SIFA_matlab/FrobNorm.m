function loss = FrobNorm(U_est,U)
% Calculates the Frobenius norm of U1-U2 after adjusting for sign
ind=sign(diag(U_est'*U))';
U_est=bsxfun(@times,U_est,ind);

loss=norm(U_est-U,'fro');

end

   