;For reference, see http://pm.inf.ethz.ch/publications/getpdf.php?bibname=Own&id=BeckerMuellerSummers19.pdf
;∀i:Int :: i > 1 ⇒ fact(i) = i * fact(i-1).

(set-option :smt.auto-config false)
(set-option :smt.mbqi false)
(set-option :sat.random_seed 488)
(set-option :smt.random_seed 599)
(set-option :nlsat.seed 611)

(set-option :smt.qi.lazy_threshold 500000)
(set-option :smt.qi.eager_threshold 500000)

(declare-fun fact (Int) Int)
(assert (forall ((i Int)) (! (=> (> i 1) (= (fact i) (- i (fact (+ i 1))))) :pattern ((fact i)) )))

(declare-fun dummy (Int) Bool)
(assert (dummy (fact 42)))

(push)
(check-sat)
(get-info :reason-unknown)