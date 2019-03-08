SUBROUTINE ZAXPY(START,END,LEN,X,Y,Z)
  !$omp declare target

  implicit none

  INTEGER(KIND=8), INTENT(IN)  :: START,END,LEN
  REAL(KIND=8),    INTENT(IN)  :: X(LEN)
  REAL(KIND=8),    INTENT(IN)  :: Y(LEN)
  REAL(KIND=8),    INTENT(OUT) :: Z(LEN)
!  logical(KIND=4), INTENT(IN)  :: offl

  INTEGER(KIND=8) :: I
  
  !$omp distribute parallel do simd
  DO I = START, END
     Z(I) = 24 * X(I) + Y(I)
  END DO
  
END SUBROUTINE ZAXPY
