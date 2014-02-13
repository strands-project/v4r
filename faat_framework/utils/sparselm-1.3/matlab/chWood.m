function x = chWood(p, n, nnz)
% nnz is unused

  for k=1:n
    i=2*(fix((k+5)/6))-1;
    switch mod(k, 6)
      case 0
        x(k)=(p(i+1)-p(i+3))/sqrt(10.0);
      case 1
        x(k)=10.0*(p(i)^2-p(i+1));
      case 2
        x(k)=p(i)-1;
      case 3
        x(k)=sqrt(90.0)*(p(i+2)^2-p(i+3));
      case 4
        x(k)=p(i+2)-1;
      case 5
        x(k)=sqrt(10.0)*(p(i+1)+p(i+3)-2);
    end
  end
