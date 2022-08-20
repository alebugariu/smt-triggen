(set-option :smt.auto-config false)
(set-option :smt.mbqi false)

(declare-sort L 0)
(declare-fun isEmpty (L) Bool)
(declare-fun contained (L Int) Bool)
(declare-fun indexOf (L Int) Int)
(declare-fun f1 (L) Int)
(declare-const EmptyList L)

(assert (! (forall ((l0 L)) (! (or (not (= l0 EmptyList)) (isEmpty l0)) :pattern ( (isEmpty l0) ))) :named A0))
(assert (! (forall ((l1 L))
    (! (or (isEmpty l1) (contained l1 (f1 l1)))
       :pattern ( (isEmpty l1) ))) :named A1))
       
(assert (! (forall ((l2 L))
    (! (or (not (isEmpty l2))
           (forall ((el2 Int)) (! (not (contained l2 el2)) :pattern ( (contained l2 el2) ))))
       :pattern ( (isEmpty l2) ))) :named A2))
(assert (! (forall ((l3 L) (el3 Int)) (! (or (contained l3 el3) (= (indexOf l3 el3) -1)) :pattern ( (contained l3 el3) ))) :named A3))
(assert (! (forall ((l4 L) (el4 Int)) (! (>= (indexOf l4 el4) 0) :pattern ( (indexOf l4 el4) ))) :named A4))

;(declare-fun dummy (Bool Bool) Bool)
;(assert (dummy (isEmpty EmptyList) (contained EmptyList 0)))

;(check-sat)