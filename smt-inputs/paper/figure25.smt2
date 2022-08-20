(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(declare-fun g (Bool Bool Bool Bool Bool Bool) Bool)
(declare-fun a () Bool)
(declare-fun b () Bool)
(declare-fun c () Bool)
(declare-fun d () Bool)
(declare-fun e () Bool)
(declare-fun f () Bool)

(assert (! (and (or (or a (and b (or c (and d e)))) f)
                (forall ((x0 Bool) (x1 Bool) (x2 Bool) (x3 Bool) (x4 Bool) (x5 Bool)) (! (g x0 x1 x2 x3 x4 x5) :pattern ( (g x0 x1 x2 x3 x4 x5))))) :named A0))

(check-sat)
