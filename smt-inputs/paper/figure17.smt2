(set-option :smt.qi.max_multi_patterns 1000)
(set-option :print-success false)
(set-info :smt-lib-version 2.0)
(set-option :AUTO_CONFIG false)
(set-option :pp.bv_literals false)
(set-option :MODEL.V2 true)
(set-option :smt.PHASE_SELECTION 0)
(set-option :smt.RESTART_STRATEGY 0)
(set-option :smt.RESTART_FACTOR |1.5|)
(set-option :smt.ARITH.RANDOM_INITIAL_VALUE true)
;(set-option :smt.CASE_SPLIT 3)
(set-option :smt.DELAY_UNITS true)
(set-option :NNF.SK_HACK true)
(set-option :smt.MBQI false)
(set-option :smt.QI.EAGER_THRESHOLD 100)
(set-option :TYPE_CHECK true)
(set-option :smt.BV.REFLECT true)
; done setting options

(set-info :category "industrial")
(declare-sort |T@U| 0)
(declare-sort |T@T| 0)
(declare-fun Ctor (T@T) Int)
(declare-fun realType () T@T)
(declare-fun boolType () T@T)
(declare-fun type (T@U) T@T)
(declare-fun RefType () T@T)
(declare-fun MapType1Type (T@T T@T) T@T)
(declare-fun MapType1TypeInv1 (T@T) T@T)
(declare-fun MapType1Select (T@U T@U T@U) T@U)
(declare-fun MapType1Store (T@U T@U T@U T@U) T@U)
(declare-fun QPMask@0 () T@U)
(declare-fun ZeroPMask () T@U)

;B0: Ctor(bool) = 2
;B1: Ctor(real) = 1
;A0: forall ((a0 T@T)(a1 T@T)) :: MapType1TypeInv1(MapType1Type(a0, a1)) = a1
;A1: forall ((a0 T@U)(a1 T@U)(a2 T@U)) :: type(MapType1Select(a0, a1, a2)) = MapType1TypeInv1(type(a0))
;A2: forall ((arg0 T@U)(arg1 T@U)(arg2 T@U)(arg3 T@U)) :: type(MapType1Store(arg0, arg1, arg2, arg3)) =
;                                                         MapType1Type(type(arg1), type(arg3))
;A3: forall ((val T@U)(m T@U)(x0 T@U)(x1 T@U)(y0 T@U)(y1 T@U)) ::
;    x0 != y0  ==> MapType1Select(MapType1Store(m, x0, x1, val) y0, y1) = MapType1Select(m, y0, y1)
;B2: type(newPMask) = MapType1Type(Ref, bool)
;B3: type(QPMask) = MapType1Type(Ref, real)

(assert (! (= (Ctor realType) 1) :named B0))
(assert (! (= (Ctor boolType) 2) :named B1))

(assert (! (forall ((arg0@@16 T@T) (arg1@@6 T@T) )
                   (! (= (MapType1TypeInv1 (MapType1Type arg0@@16 arg1@@6)) arg1@@6)
           :pattern ( (MapType1Type arg0@@16 arg1@@6)))) :named A0))

(assert (! (forall ((arg0@@17 T@U) (arg1@@7 T@U) (arg2@@1 T@U) )
           (! (let ((aVar1 (MapType1TypeInv1 (type arg0@@17))))
              (= (type (MapType1Select arg0@@17 arg1@@7 arg2@@1)) aVar1))
           :pattern ( (MapType1Select arg0@@17 arg1@@7 arg2@@1)))) :named A1))

(assert (! (forall ((arg0@@18 T@U) (arg1@@8 T@U) (arg2@@2 T@U) (arg3@@0 T@U) )
           (! (let ((aVar1@@0 (type arg3@@0))) (let ((aVar0@@0 (type arg1@@8)))
              (= (type (MapType1Store arg0@@18 arg1@@8 arg2@@2 arg3@@0)) (MapType1Type aVar0@@0 aVar1@@0))))
           :pattern ( (MapType1Store arg0@@18 arg1@@8 arg2@@2 arg3@@0)))) :named A2))

(assert (! (forall ((val@@4 T@U) (m@@4 T@U) (x0@@4 T@U) (x1@@4 T@U) (y0@@2 T@U) (y1@@2 T@U) )
           (!  (or (= x0@@4 y0@@2) (= (MapType1Select (MapType1Store m@@4 x0@@4 x1@@4 val@@4) y0@@2 y1@@2) (MapType1Select m@@4 y0@@2 y1@@2)))
           :pattern ( (MapType1Select (MapType1Store m@@4 x0@@4 x1@@4 val@@4) y0@@2 y1@@2)))) :named A3))

(assert (! (= (type ZeroPMask) (MapType1Type RefType boolType)) :named B2))
(assert (! (= (type QPMask@0) (MapType1Type RefType realType)) :named B3))

;(declare-fun __dummyA3__ (T@U) Bool)

;(assert (__dummyA3__ (MapType1Select (MapType1Store (MapType1Select QPMask@0 ZeroPMask ZeroPMask) QPMask@0 ZeroPMask ZeroPMask) ZeroPMask ZeroPMask)))
;(assert (__dummyA3__ (MapType1Select (MapType1Store (MapType1Select QPMask@0 ZeroPMask ZeroPMask) ZeroPMask QPMask@0 QPMask@0) QPMask@0 QPMask@0)))

(check-sat)

;Solved by: Optset(top_level=False)