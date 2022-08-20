(set-option :AUTO_CONFIG false)
(set-option :smt.MBQI false)

(set-info :category "industrial")
(declare-sort |T@U| 0)
(declare-sort |T@T| 0)
(declare-fun Ctor (T@T) Int)
(declare-fun intType () T@T)
(declare-fun boolType () T@T)
(declare-fun type (T@U) T@T)
(declare-fun RefType () T@T)
(declare-fun int_2_U (Int) T@U)
(declare-fun bool_2_U (Bool) T@U)
(declare-fun U_2_int (T@U) Int)
(declare-fun MapType1Type (T@T T@T) T@T)
(declare-fun MapType1TypeInv1 (T@T) T@T) ; inverse function
(declare-fun MapType1Select (T@U T@U T@U) T@U)
(declare-fun MapType1Store (T@U T@U T@U T@U) T@U)

;B0: Ctor(bool) = 2
;B1: Ctor(real) = 1
;A0: forall ((a0 T@T)(a1 T@T)) :: MapType1TypeInv1(MapType1Type(a0, a1)) = a1 ; def. of inverse for range type
;A1: forall ((a0 T@U)(a1 T@U)(a2 T@U)) :: type(MapType1Select(a0, a1, a2)) = MapType1TypeInv1(type(a0)) ; select gives value of range type
;A2: forall ((arg0 T@U)(arg1 T@U)(arg2 T@U)(arg3 T@U)) :: type(MapType1Store(arg0, arg1, arg2, arg3)) =
;                                                         MapType1Type(type(arg1), type(arg3)) ; store yields mask from index type to rhs type, irrespective of the original map type!
;A3: forall ((val T@U)(m T@U)(x0 T@U)(x1 T@U)(y0 T@U)(y1 T@U)) ::
;    x0 != y0  ==> MapType1Select(MapType1Store(m, x0, x1, val) y0, y1) = MapType1Select(m, y0, y1) ; map entries at different positions are independent
;A4: forall ((a0 T@T)) :: type(int_2_U(a0)) = intType
;A5: forall ((a0 T@T)) :: type(bool_2_U(a0)) = boolType
;A6: forall ((a0 T@T)) :: U_2_int(int_2_U(a0)) = a0

(assert (! (= (Ctor intType) 1) :named B0))
(assert (! (= (Ctor boolType) 2) :named B1))

(assert (! (forall ((kt0 T@T) (vt0 T@T) )
                   (! (= (MapType1TypeInv1 (MapType1Type kt0 vt0)) vt0)
           :pattern ( (MapType1Type kt0 vt0)))) :named A0))

(assert (! (forall ((m1 T@U) (k1 T@U) (v1 T@U) )
           (! (let ((aVar1 (MapType1TypeInv1 (type m1))))
              (= (type (MapType1Select m1 k1 v1)) aVar1))
           :pattern ( (MapType1Select m1 k1 v1)))) :named A1))

(assert (! (forall ((m2 T@U) (k2 T@U) (x2 T@U) (v2 T@U) )
           (! (let ((aVar1@@0 (type v2))) (let ((aVar0@@0 (type k2)))
              (= (type (MapType1Store m2 k2 x2 v2)) (MapType1Type aVar0@@0 aVar1@@0))))
           :pattern ( (MapType1Store m2 k2 x2 v2)))) :named A2))

(assert (! (forall ((v3 T@U) (m3 T@U) (k3 T@U) (x3 T@U) (other_k3 T@U) (other_v3 T@U) )
           (!  (or (= k3 other_k3) (= (MapType1Select (MapType1Store m3 k3 x3 v3) other_k3 other_v3) (MapType1Select m3 other_k3 other_v3)))
           :pattern ( (MapType1Select (MapType1Store m3 k3 x3 v3) other_k3 other_v3)))) :named A3))

(assert (! (forall ((arg4 Int) ) (! (= (type (int_2_U arg4)) intType) :pattern ( (int_2_U arg4)))) :named A4))
(assert (! (forall ((arg5 Bool) ) (! (= (type (bool_2_U arg5)) boolType) :pattern ( (bool_2_U arg5)))) :named A5))
(assert (! (forall ((arg6 Int) ) (! (= (U_2_int (int_2_U arg6)) arg6) :pattern ( (int_2_U arg6)))) :named A6))

;(declare-fun __dummyA3__ (T@U) Bool)
;(declare-fun m () T@U)
;(declare-fun x () T@U)

;(assert (__dummyA3__ (MapType1Select (MapType1Store (MapType1Store m (int_2_U 0) x (bool_2_U true))
;                                     (int_2_U 1) x (int_2_U 0)) (int_2_U 0) x)))

(check-sat)

;Solved by: Optset(top_level=False)