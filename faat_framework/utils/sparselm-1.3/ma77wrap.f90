! HSL MA77 wrapper
! Manolis Lourakis, October 2010

! Sample code to illustrate invocation of ma77_wrapper (remove !!'s)
!!program ma77_wrap
!!      implicit none
!!! Parameters
!!      integer, parameter :: wp = kind(0.0d0)
!!! WARNING: these should not be hardcoded!
!!!     integer, parameter :: n=5, nnz=13
!!      integer, parameter :: n=5, nnz=9
!!
!!      integer :: ret
!!      character(len=20) :: fname(4)
!!! full sparse matrix in CCS
!!!     integer :: rowidx(nnz) = &
!!!         (/1, 2, 1, 2, 3, 5, 2, 3, 4, 3, 4, 2, 5/)
!!!     integer :: colptr(n+1) = &
!!!         (/ 1, 3, 7, 10, 12, 14 /)
!!!     real(wp) :: val(nnz) = &
!!!         (/2., 3., 3., 1., 4., 6., 4., 1., 5., 5., 3., 6., 1./)
!!
!!! sparse matrix lower triangle in CCS
!!      integer :: rowidx(nnz) = &
!!          (/1, 2,   2, 3, 5,   3, 4,   4,   5/)
!!      integer :: colptr(n+1) = &
!!          (/ 1, 3, 6, 8, 9, 10 /)
!!      real(wp) :: val(nnz) = &
!!          (/2., 3.,   1., 4., 6.,   1., 5.,   3.,   1./)
!!
!!
!!      real(wp) :: x(n, 1)
!!      data x/5., 14., 10., 8., 7./
!!
!!      interface
!!        integer function ma77_wrapper(what, n,filenames, nnz,colptr,rowidx,val, x)
!!        implicit none
!!        integer, parameter :: wp = kind(0.0d0)
!!        integer, intent (in) :: what, n
!!        character(len=20) :: filenames(4)
!!        integer :: nnz
!!        integer, dimension (*) :: colptr, rowidx
!!        real(wp), dimension (*) :: val
!!        real(wp), dimension (n,1) :: x
!!        end function
!!
!!        integer function ma77_wrapper_genrow(j, colptr, rowidx, val, row, cols)
!!        implicit none
!!        integer, parameter :: wp = kind(0.0d0)
!!        integer :: j
!!        integer, dimension (*) :: colptr, rowidx
!!        real(wp), dimension (*) :: val
!!        real(wp), dimension (*) :: row
!!        integer, dimension (*) :: cols
!!        end function
!!      end interface
!!
!!! Choose file indentifiers (hold direct access files in current directory)
!!      fname(1) = 'factor_integer'
!!      fname(2) = 'factor_real'
!!      fname(3) = 'work_real'
!!      fname(4) = 'temp1'
!!
!!      print *, 'starting... '
!!! 1: Allocate, initialize, factorize symbolically
!!      ret=ma77_wrapper(1, n,fname, nnz,colptr,rowidx,val, x)
!!
!!! 2: Load array elements, factorize numerically, solve
!!      ret=ma77_wrapper(2, n,fname, nnz,colptr,rowidx,val, x)
!!
!!! 3: Deallocate
!!      ret=ma77_wrapper(3, n,fname, nnz,colptr,rowidx,val, x)
!!      write (*,'(/a,/,(6f10.3))') ' The computed solution is:',x(1:n,1:1)
!!end program ma77_wrap


! wrapper function code
integer function ma77_wrapper(what, n,filenames, nnz,colptr,rowidx,val, x)
use hsl_MA77_double
use hsl_zd11_double
use hsl_mc68_double
implicit none
integer, parameter :: wp = kind(0.0d0)
integer, intent (in) :: what, n
character(len=*), intent (in) :: filenames(4)
integer, intent (in) :: nnz
integer, dimension (*), intent (in) :: colptr, rowidx
real(wp), dimension (*), intent (in) :: val
real(wp), dimension (n,1), intent (inout) :: x

! Derived types
type (ma77_keep), save, pointer    :: keep
type (ma77_control), save, pointer :: control77
type (ma77_info), save, pointer    :: info77

! Following three variables are needed for computing a pivot order
type (mc68_control) :: control68
type (mc68_info) :: info68
type (zd11_type) :: a

integer, dimension (:),  save, allocatable :: order, cols
real(wp), dimension (:), save, allocatable :: row

! real(wp), dimension (:,:), save, allocatable :: resid
! integer :: lresid
integer :: i, lx, nrhs, nvar, ret=0, ma77_wrapper_genrow
logical :: pos_def

!     print *, 'What= ', what, n
!     print*,'filenames(1,2,3,4) =',filenames(1),',',filenames(2),',',filenames(3),',',filenames(4),','

      select case (what)
        case (1)
! Allocate arrays of appropriate size
          allocate (order(n), cols(n), row(n))

! Allocate derived types
          allocate(keep)
          allocate(control77)
          allocate(info77)

! Initialize the data structures and open the superfiles
!         control77%nemin = 1 ! node amalgamation control
          !control77%maxstore = 22937600 ! max storage for using in-core arrays; 700Mb on 32 systems
          !control77%storage(1)=xxxx
          !control77%storage(2)=xxxx
          !control77%storage(3)=xxxx
          !control77%storage(4)=0

          call ma77_open(n, filenames, keep, control77, info77)
          if (info77%flag < 0) then
            ret=info77%flag ! save
            call ma77_finalise(keep, control77, info77)
            info77%flag=ret
            go to 100
          end if

! Since the matrix is symmetric, its rows equal its columns.
! For each column of the matrix, supply the number of elements
! and the row indices
          do i = 1,n
            nvar=ma77_wrapper_genrow(i, colptr, rowidx, val, row, cols)
            ! Specify which variables are associated with each row
            call ma77_input_vars(i, nvar, cols, keep, control77, info77)
            if (info77%flag < 0) then
              ret=info77%flag ! save
              call ma77_finalise(keep, control77, info77)
              info77%flag=ret
              go to 100
            end if
          end do

! Use the natural pivot order 1,2,...,n
!         do i = 1,n
!           order(i) = i
!         end do

! Compute a pivot order
! Set up data using mc68_setup
          call mc68_setup(a, n, nnz, rowidx, control68, info68, ptr=colptr)

! Compute elimination order using the approximate minimum degree method
          call mc68_order(1, a, order, control68, info68)
!         write (6,'(a)') ' Approximate minimum degree ordering : '
!         write (6,'(8i4)') order
!         write (6,'(a)') ' '

! Compute elimination order using the MA47 method
!         call mc68_order(4, a, order, control68, info68)
!         write (6,'(a)') ' MA47 ordering : '
!         write (6,'(8i4)') order

          call mc68_finalize(a, control68, info68)

! Perform analyse
          call ma77_analyse(order, keep, control77, info77)
          if (info77%flag < 0) go to 100
          go to 200

        case (2)
! Load numerical values
          do i = 1,n
            nvar=ma77_wrapper_genrow(i, colptr, rowidx, val, row, cols)
            ! Specify the entries of each row
            call ma77_input_reals(i, nvar, row, keep, control77, info77)
            if (info77%flag < 0) go to 100
          end do

          ! compute scaling factors
          !call MA77_scale(scale, keep, control77, info77)

! Copy the right-hand side into resid
!         allocate (resid(1:n,1:1))
!         resid(1:n,1:1) = x(1:n,1:1)
! Factorise and solve using the computed factors 
! ma77_factor_solve should be more efficient than ma77_factor, ma77_solve 
          !pos_def = .false.
          pos_def = .true.
          nrhs = 1
          lx = size(x, 1)
          call ma77_factor_solve(pos_def, keep, control77, info77, nrhs, lx, x)
          if (info77%flag < 0) go to 100
          ! storage is expressed in Fortran storage units below;
          ! should be multiplied by sizeof(int) to convert to bytes
          !write (*,'(a,i15)') 'ma77_wrapper(): storage used in superfiles = ', info77%minstore
          ! numbers of integers and reals stored in superfiles
          !write (*,'(a,i10,i10,i10,i10)') 'ma77_wrapper(): storage used per superfile ', &
          !info77%storage(1), info77%storage(2), info77%storage(3), info77%storage(4)


! Compute the residuals
!         lresid = size(resid,1)
!         call ma77_resid(nrhs, lx, x, lresid, resid, keep, control77, info77)
!         if (info77%flag < 0) go to 100
!         write (*,'(/a,/,(6f10.3))') ' The computed solution is:', x(1:n,1:1)
!         write (*,'(/a,/,(6f10.3))') ' The residuals are:', resid(1:n,1:1)
!         deallocate (resid)
          go to 200

        case (3)
! Deallocate
          call ma77_finalise(keep, control77, info77)
          go to 300

        case default
          print *, 'ma77_wrapper(): unknown argument "what" = ', what
          ret=1
          go to 200
        end select

! Print error message
100     ret=1 ! error
        write (*,'(a,i3)') 'ma77_wrapper() error. info%flag = ', info77%flag

! Deallocate all arrays
300     deallocate (order, cols, row)
        deallocate (keep)
        deallocate (control77)
        deallocate (info77)

! Return
200     ma77_wrapper=ret
end function ma77_wrapper



! Generate the full row j of a sparse symmetric matrix given its
! diagonal and lower triangle in CCS format
! Fills in the row & cols vectors with the values and column indices
! of the nonzero elements in row j and returns their number
integer function ma77_wrapper_genrow(j, colptr, rowidx, val, row, cols)
implicit none
integer, parameter :: wp = kind(0.0d0)
integer, intent(in) :: j
integer, intent(in), dimension (*) :: colptr, rowidx
real(wp), intent(in), dimension (*) :: val
real(wp), intent(out), dimension (*) :: row
integer, intent(out), dimension (*) :: cols

integer :: i, k, ii
integer :: high, low, mid, diff

      k=0
! first, find any non-zero elements in the upper triangle. This
! is equivalent to finding all elements on row j at columns i<j
upr:  do i=1, j-1
        mid=0
        low=colptr(i)
        high=colptr(i+1)-1

! binary search for finding an element in row j and column i
        do while(low<=high)
          mid=(low+high)/2
          diff=j-rowidx(mid)
          if(diff<0) then
            high=mid-1
          else if(diff>0) then
            low=mid+1
          else
            ! found a nonzero element at (i, j)
            k=k+1
            cols(k)=i ! row becomes column in the upper (i.e., transposed lower) triangle
            row(k)=val(mid)
            cycle upr
          end if
        end do

      ! this point is reached on search failure 
      end do upr

! then, consider all elements of column j in the diagonal and lower triangle
      low=colptr(j)
      high=colptr(j+1)-1
lwr:  do i=low, high
        if(rowidx(i)>=j) then
          do ii=i, high
            k=k+1
            cols(k)=rowidx(ii)
            row(k)=val(ii)
          end do
          exit lwr
        end if
      end do lwr

      ma77_wrapper_genrow=k
end function 
