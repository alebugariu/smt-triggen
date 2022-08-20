(declare-fun $div (Int Int) Int)
(declare-fun z3name!0 (Int Int) Int)
(declare-fun $sign (Int) Int)

(assert (forall ((a!1 Int) (b!1 Int))
          (! (= (to_real ($div a!1 b!1))
                (+ (/ (to_real a!1) (to_real b!1)) (to_real (z3name!0 b!1 a!1))))
             :pattern (($div a!1 b!1))
             :pattern ((to_real b!1) (to_real (z3name!0 b!1 a!1))) )))
