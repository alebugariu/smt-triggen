(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun both_ptr (Int Int Int) Int)

(assert (! (forall ((a Int) (b Int) (size Int)) (! (<= (* (both_ptr a b size) size) (+ a (* (- 1) b))) :pattern ( (both_ptr a b size)))) :named A0))

(check-sat)