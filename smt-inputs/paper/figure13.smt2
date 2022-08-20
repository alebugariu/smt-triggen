(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun P (Int) Bool)
(declare-fun Q (Int Int Int) Bool)
(declare-fun R (Int Int) Bool)
(assert (forall ((z0 Int)) (!
  (or
    (not (P z0))
    (forall ((x0 Int) (y0 Int)) (!
        (Q x0 y0 z0)
     :pattern ((Q x0 y0 z0)) )))
 :pattern ((P z0)))))

(assert (forall ((z0 Int)) (!
  (or
    (not (P z0))
    (forall ((x0 Int) (y0 Int)) (!
        (R y0 z0)
     :pattern ((Q x0 y0 z0)) )))
 :pattern ((P z0)))))

(assert (forall ((z1 Int)) (! (P z1) :pattern ((P z1)))))
(assert (forall ((y2 Int) (z2 Int)) (! (not (R y2 z2)) :pattern ((R y2 z2)))))


;(declare-fun __dummy__ (Bool Bool) Bool)
;(assert (__dummy__ (P 0) (Q 0 0 0)))
;(check-sat)

