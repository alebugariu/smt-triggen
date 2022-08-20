(set-option :smt.auto-config false)
(set-option :smt.mbqi false)

(declare-sort ISeq 0)

(declare-fun sum (ISeq Int Int) Int)
(declare-fun sum_syn (ISeq Int Int) Int)
(declare-fun empty () ISeq)
(declare-fun seq.nth (ISeq Int) Int)

(assert (! (forall ((xs0 ISeq) (l0 Int) (h0 Int)) (! (= (sum xs0 l0 h0) (sum_syn xs0 l0 h0)) :pattern ((sum xs0 l0 h0)) )) :named A0 ))
(assert (! (forall ((xs1 ISeq) (l1 Int) (h1 Int)) (!
    (or 
        (not (>= l1 h1))
        (= (sum_syn xs1 l1 h1) 0))
     :pattern ((sum xs1 l1 h1)) )) :named A1 ))
     
(assert (! (forall ((xs2 ISeq) (l2 Int) (h2 Int)) (!
    (or 
        (not (<= l2 h2))
        (= 
            (sum_syn xs2 l2 h2)
            (+ 
                (sum_syn xs2 (+ l2 1) h2)
                (seq.nth xs2 l2) )))
     :pattern ((sum xs2 l2 h2)) )) :named A2 ))
      
(assert (! (= (seq.nth empty 0) -1) :named A3 ))

;(declare-fun dummy (Int Int) Bool)
;(assert (dummy (sum empty 0 0) (sum empty 1 0)))

;(check-sat)
