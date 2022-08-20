(set-option :smt.auto-config false) ; disable automatic self configuration
(set-option :smt.mbqi false) ; disable model-based quantifier instantiation

(declare-sort iList 0)
(declare-fun empty () iList)
(declare-fun cons (Int iList) iList)
(declare-fun len (iList) Int)

; A0
(assert (forall ((x Int) (xs iList)) (! (= (len (cons x xs)) (+ (len xs) 1)) :pattern ((cons x xs)))))

; A1
(assert (forall ((y Int) (es iList)) (!
    (=>
        (= es empty)
        (= (len (cons y es)) 0))

:pattern ((cons y es)))))

; A2
(assert (forall ((zs iList)) (! (>= (len zs) 0) :pattern ((len zs)))))

;;;;; Solution:
;(declare-fun dummy (iList) Bool)
;(assert (dummy (cons 42 empty)))
;(check-sat)

; skolem_neg(A0)
;(declare-fun xs!0 () iList)
;(declare-fun x!1 () Int)
;(assert (not (= (len (cons x!1 xs!0)) (+ (len xs!0) 1))))

;(apply (using-params nnf :mode quantifiers) :print false :print_benchmark true)
;(check-sat)
;(get-value (x!1 xs!0))
;((x!1 2) (xs!0 iList!val!0))

;(get-value (x!1 xs!0 empty))
;((x!1 2) (xs!0 iList!val!0) (empty iList!val!0))