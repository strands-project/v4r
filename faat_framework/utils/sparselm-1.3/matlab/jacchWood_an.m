% demonstrates two ways of supplying a sparse Jacobian,
% i.e. directly and through (i, j, val) triplets

function jac = jacchWood_an(p, n, nnz)
  jac=jacchWood_dir(p, n, nnz);
  %jac=jacchWood_tri(p, n, nnz);


% the Jacobian is directly created
function jac = jacchWood_dir(p, n, nnz)
  m=max(size(p));

  % create an empty sparse matrix
  jac=sparse([], [], [], n, m, nnz);
  for k=1:n
    i=2*(fix((k+5)/6))-1;
    % fill in nonzero elements
    switch mod(k, 6)
      case 0
        jac(k, i+1)=1.0/sqrt(10.0);
        jac(k, i+3)=-1.0/sqrt(10.0);
      case 1
        jac(k, i)=20.0*p(i);
        jac(k, i+1)=-10.0;
      case 2
        jac(k, i)=1.0;
      case 3
        jac(k, i+2)=2.0*sqrt(90.0)*p(i+2);
        jac(k, i+3)=-sqrt(90.0);
      case 4
        jac(k, i+2)=1.0;
      case 5
        jac(k, i+1)=sqrt(10.0);
        jac(k, i+3)=sqrt(10.0);
    end
  end


% alternatively, the Jacobian is first created in sparse
% triplet format and then converted to a matlab sparse matrix (CCS)
function jac = jacchWood_tri(p, n, nnz)
  m=max(size(p));

  % preallocate triplet vectors
  rowidx=zeros(nnz, 1);
  colidx=zeros(nnz, 1); 
  val=zeros(nnz, 1); 
  l=1;
  for k=1:n
    i=2*(fix((k+5)/6))-1;
    % supply the row, column indices and values of nonzero Jacobian elements
    switch mod(k, 6)
      case 0
        val(l)=1.0/sqrt(10.0);  rowidx(l)=k; colidx(l)=i+1; l=l+1;
        val(l)=-1.0/sqrt(10.0); rowidx(l)=k; colidx(l)=i+3; l=l+1;
      case 1
        val(l)=20.0*p(i); rowidx(l)=k; colidx(l)=i; l=l+1;
        val(l)=-10.0; rowidx(l)=k;     colidx(l)=i+1; l=l+1;
      case 2
        val(l)=1.0; rowidx(l)=k; colidx(l)=i; l=l+1;
      case 3
        val(l)=2.0*sqrt(90.0)*p(i+2); rowidx(l)=k; colidx(l)=i+2; l=l+1;
        val(l)=-sqrt(90.0);           rowidx(l)=k; colidx(l)=i+3; l=l+1;
      case 4
        val(l)=1.0; rowidx(l)=k; colidx(l)=i+2; l=l+1;
      case 5
        val(l)=sqrt(10.0); rowidx(l)=k; colidx(l)=i+1; l=l+1;
        val(l)=sqrt(10.0); rowidx(l)=k; colidx(l)=i+3; l=l+1;
    end
  end

  % convert to sparse matrix
  jac=sparse(rowidx, colidx, val, n, m, l);
