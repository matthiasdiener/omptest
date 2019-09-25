SUBROUTINE ZAXPY(START,END,LEN,X,Y,Z)
  !$omp declare target

  INTEGER(KIND=8), INTENT(IN)  :: START,END,LEN
  REAL(KIND=8),    INTENT(IN)  :: X(LEN)
  REAL(KIND=8),    INTENT(IN)  :: Y(LEN)
  REAL(KIND=8),    INTENT(OUT) :: Z(LEN)

  INTEGER(KIND=8) :: I
  
  !$omp distribute parallel do simd
  DO I = START, END
     Z(I) = 24 * X(I) + Y(I)
  END DO
  
END SUBROUTINE ZAXPY


!ICE test - base function
SUBROUTINE ZAXPY_BASE(START,END,LEN,X,Y,Z,A)

  INTEGER(KIND=8), INTENT(IN)  :: START,END,LEN
  REAL(KIND=8),    INTENT(IN)  :: X(LEN),A
  REAL(KIND=8),    INTENT(IN)  :: Y(LEN)
  REAL(KIND=8),    INTENT(OUT) :: Z(LEN)

  INTEGER(KIND=8) :: I

  DO I = START, END
    Z(I) = A * X(I) + Y(I)
  END DO

END SUBROUTINE ZAXPY_BASE


!ICE test - offloaded function
SUBROUTINE ZAXPY_MOD(START,END,LEN,X,Y,Z,A)

  INTEGER(KIND=8), INTENT(IN)  :: START,END,LEN
  REAL(KIND=8),    INTENT(IN)  :: X(LEN),A
  REAL(KIND=8),    INTENT(IN)  :: Y(LEN)
  REAL(KIND=8),    INTENT(OUT) :: Z(LEN)

  INTEGER(KIND=8) :: I

  !$omp target teams distribute parallel do simd is_device_ptr(X,Y,Z) num_teams(80) thread_limit(2000)
  DO I = START, END
    Z(I) = A * X(I) + Y(I)
  END DO

END SUBROUTINE ZAXPY_MOD
