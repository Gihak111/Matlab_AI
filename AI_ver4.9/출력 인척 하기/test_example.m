A = rand(1, 10)
reshape(A, 5, 2)

reshape(A, 2, 5)

reshape(A,5,3)   % error, becuase A has only 10 elements and you are trying to reshape 10 elements into 5*3 = 15
