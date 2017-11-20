SUBROUTINE ZAXPY(START,END,LEN,X,Y,Z)

  INTEGER(KIND=8), INTENT(IN)  :: START,END,LEN
  REAL(KIND=8),    INTENT(IN)  :: X(LEN)
  REAL(KIND=8),    INTENT(IN)  :: Y(LEN)
  REAL(KIND=8),    INTENT(OUT) :: Z(LEN)

  INTEGER(KIND=8) :: I
  
  !$acc parallel loop
  DO I = START, END
     Z(I) = 24 * X(I) + Y(I)
  END DO
  !$acc end parallel loop
  
END SUBROUTINE ZAXPY
