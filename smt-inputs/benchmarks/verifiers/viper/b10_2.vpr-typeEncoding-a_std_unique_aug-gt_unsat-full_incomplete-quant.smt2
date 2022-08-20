(set-info :smt-lib-version 2.6)
(set-info :category "industrial")
(set-info :boogie-vc-id Triggerstriggers_three$)
(set-option :smt.auto-config false)
(set-option :smt.mbqi false)
(set-option :sat.random_seed 488)
(set-option :smt.random_seed 599)
(set-option :nlsat.seed 611)
(set-option :memory_max_size 6000)
(declare-sort |T@U| 0)
(declare-sort RegExStr 0)
(declare-sort RMode 0)
(declare-sort |T@T| 0)
(declare-fun real_pow (Real Real) Real)
(declare-fun UOrdering2 (|T@U| |T@U|) Bool)
(declare-fun UOrdering3 (|T@T| |T@U| |T@U|) Bool)
(declare-fun tickleBool (Bool) Bool)
(declare-fun Ctor (T@T) Int)
(declare-fun intType () T@T)
(declare-fun realType () T@T)
(declare-fun boolType () T@T)
(declare-fun rmodeType () T@T)
(declare-fun stringType () T@T)
(declare-fun regexType () T@T)
(declare-fun int_2_U (Int) T@U)
(declare-fun U_2_int (T@U) Int)
(declare-fun real_2_U (Real) T@U)
(declare-fun U_2_real (T@U) Real)
(declare-fun bool_2_U (Bool) T@U)
(declare-fun U_2_bool (T@U) Bool)
(declare-fun rmode_2_U (RMode) T@U)
(declare-fun U_2_rmode (T@U) RMode)
(declare-fun string_2_U (String) T@U)
(declare-fun U_2_string (T@U) String)
(declare-fun regex_2_U (RegExStr) T@U)
(declare-fun U_2_regex (T@U) RegExStr)
(declare-fun MapType0Select (T@T T@T T@T T@U T@U T@U) T@U)
(declare-fun NormalFieldType () T@T)
(declare-fun RefType () T@T)
(declare-fun $allocated () T@U)
(declare-fun MapType0Store (T@T T@T T@T T@U T@U T@U T@U) T@U)
(declare-fun IdenticalOnKnownLocations (T@U T@U T@U) Bool)
(declare-fun HasDirectPerm (T@T T@T T@U T@U T@U) Bool)
(declare-fun IsPredicateField (T@T T@T T@U) Bool)
(declare-fun FrameTypeType () T@T)
(declare-fun MapType1Type (T@T T@T) T@T)
(declare-fun null () T@U)
(declare-fun PredicateMaskField (T@T T@U) T@U)
(declare-fun MapType1Select (T@T T@T T@T T@T T@U T@U T@U) T@U)
(declare-fun MapType1Store (T@T T@T T@T T@T T@U T@U T@U T@U) T@U)
(declare-fun MapType1TypeInv0 (T@T) T@T)
(declare-fun MapType1TypeInv1 (T@T) T@T)
(declare-fun IsWandField (T@T T@T T@U) Bool)
(declare-fun WandMaskField (T@T T@U) T@U)
(declare-fun succHeap (T@U T@U) Bool)
(declare-fun succHeapTrans (T@U T@U) Bool)
(declare-fun ZeroMask () T@U)
(declare-fun NoPerm () Real)
(declare-fun ZeroPMask () T@U)
(declare-fun FullPerm () Real)
(declare-fun state (T@U T@U) Bool)
(declare-fun GoodMask (T@U) Bool)
(declare-fun sumMask (T@U T@U T@U) Bool)
(declare-fun ConditionalFrame (Real T@U) T@U)
(declare-fun EmptyFrame () T@U)
(declare-fun InsidePredicate (T@T T@T T@U T@U T@U T@U) Bool)
(declare-fun Triggersf$ (T@U T@U Int Int Int) Bool)
(declare-fun |Triggersf$@quote@| (T@U T@U Int Int Int) Bool)
(declare-fun dummyFunction (T@T T@U) Bool)
(declare-fun |Triggersf$@sharp@triggerStateless| (T@U Int Int Int) Bool)
(declare-fun |Triggersf$@sharp@frame| (T@U T@U Int Int Int) Bool)
(declare-fun PredicateType_Triggersvalid$Type () T@T)
(declare-fun Triggersvalid$ (T@U) T@U)
(declare-fun Triggersg$ (T@U T@U Int) Bool)
(declare-fun |Triggersg$@quote@| (T@U T@U Int) Bool)
(declare-fun |Triggersg$@sharp@triggerStateless| (T@U Int) Bool)
(declare-fun |Triggersg$@sharp@frame| (T@U T@U Int) Bool)
(declare-fun |Triggersvalid$@sharp@sm| (T@U) T@U)
(declare-fun getPredicateId (T@T T@T T@U) Int)
(declare-fun |Triggersvalid$@sharp@trigger| (T@T T@U T@U) Bool)
(declare-fun |Triggersvalid$@sharp@everUsed| (T@T T@U) Bool)
(declare-fun ControlFlow (Int Int) Int)
(declare-fun this$_8 () T@U)
(declare-fun wildcard@0 () Real)
(declare-fun perm@1 () Real)
(declare-fun PostMask@0 () T@U)
(declare-fun PostHeap@0 () T@U)
(declare-fun wildcard@1 () Real)
(declare-fun perm@2 () Real)
(declare-fun Heap@@16 () T@U)
(declare-fun w_2 () Int)
(declare-fun x_2 () Int)
(declare-fun y_2 () Int)
(declare-fun z_2 () Int)
(declare-fun Mask@1 () T@U)
(declare-fun Mask@0 () T@U)
(declare-fun perm@0 () Real)
(declare-fun n$_2 () T@U)
(declare-fun Mask@@11 () T@U)
(declare-fun wildcard@3 () Real)
(declare-fun perm@4 () Real)
(declare-fun a_2 () Int)
(declare-fun wildcard@2 () Real)
(declare-fun perm@3 () Real)
(declare-fun AssumeFunctionsAbove () Int)
(declare-fun k$_2 () Real)
(declare-fun wildcard () Real)
(declare-fun z3name!0 (T@U Real) T@U)
(assert (! (tickleBool false) :named A0_0))
(assert (! (tickleBool true) :named A0_1))
(assert (! (= (Ctor boolType) 2) :named A1_0))
(assert (! (= (Ctor intType) 0) :named A1_1))
(assert (! (= (Ctor realType) 1) :named A1_2))
(assert (! (= (Ctor regexType) 5) :named A1_3))
(assert (! (= (Ctor rmodeType) 3) :named A1_4))
(assert (! (= (Ctor stringType) 4) :named A1_5))
(assert (! (forall ((arg0py0 Int)) (! (= (U_2_int (int_2_U arg0py0)) arg0py0) :pattern ((int_2_U arg0py0)) )) :named A1_6))
(assert (! (forall ((arg0@@0py0 Real)) (! (= (U_2_real (real_2_U arg0@@0py0)) arg0@@0py0) :pattern ((real_2_U arg0@@0py0)) )) :named A1_7))
(assert (! (forall ((arg0@@1py0 Bool)) (! (and (or (not (U_2_bool (bool_2_U arg0@@1py0))) arg0@@1py0) (or (U_2_bool (bool_2_U arg0@@1py0)) (not arg0@@1py0))) :pattern ((bool_2_U arg0@@1py0)) )) :named A1_8))
(assert (! (forall ((arg0@@2py0 RMode)) (! (= (U_2_rmode (rmode_2_U arg0@@2py0)) arg0@@2py0) :pattern ((rmode_2_U arg0@@2py0)) )) :named A1_9))
(assert (! (forall ((arg0@@3py0 String)) (! (= (U_2_string (string_2_U arg0@@3py0)) arg0@@3py0) :pattern ((string_2_U arg0@@3py0)) )) :named A1_10))
(assert (! (forall ((arg0@@4py0 RegExStr)) (! (= (U_2_regex (regex_2_U arg0@@4py0)) arg0@@4py0) :pattern ((regex_2_U arg0@@4py0)) )) :named A1_11))
(assert (! (forall ((xpy0 T@U)) (! (= (int_2_U (U_2_int xpy0)) xpy0) :pattern ((U_2_int xpy0)) )) :named A1_12))
(assert (! (forall ((x@@0py0 T@U)) (! (= (real_2_U (U_2_real x@@0py0)) x@@0py0) :pattern ((U_2_real x@@0py0)) )) :named A1_13))
(assert (! (forall ((x@@1py0 T@U)) (! (= (bool_2_U (U_2_bool x@@1py0)) x@@1py0) :pattern ((U_2_bool x@@1py0)) )) :named A1_14))
(assert (! (forall ((x@@2py0 T@U)) (! (= (rmode_2_U (U_2_rmode x@@2py0)) x@@2py0) :pattern ((U_2_rmode x@@2py0)) )) :named A1_15))
(assert (! (forall ((x@@3py0 T@U)) (! (= (string_2_U (U_2_string x@@3py0)) x@@3py0) :pattern ((U_2_string x@@3py0)) )) :named A1_16))
(assert (! (forall ((x@@4py0 T@U)) (! (= (regex_2_U (U_2_regex x@@4py0)) x@@4py0) :pattern ((U_2_regex x@@4py0)) )) :named A1_17))
(assert (! (forall ((x@@5py0 T@U)(alphapy0 T@T)) (! (UOrdering3 alphapy0 x@@5py0 x@@5py0) :pattern ((UOrdering3 alphapy0 x@@5py0 x@@5py0)) )) :named A2))
(assert (! (forall ((x@@6py0 T@U)(ypy0 T@U)(zpy0 T@U)(alpha@@0py0 T@T)) (! (or (UOrdering3 alpha@@0py0 x@@6py0 zpy0) (not (UOrdering3 alpha@@0py0 x@@6py0 ypy0)) (not (UOrdering3 alpha@@0py0 ypy0 zpy0))) :pattern ((UOrdering3 alpha@@0py0 x@@6py0 ypy0) (UOrdering3 alpha@@0py0 ypy0 zpy0)) )) :named A3))
(assert (! (forall ((x@@7py0 T@U)(y@@0py0 T@U)(alpha@@1py0 T@T)) (! (or (= x@@7py0 y@@0py0) (not (UOrdering3 alpha@@1py0 x@@7py0 y@@0py0)) (not (UOrdering3 alpha@@1py0 y@@0py0 x@@7py0))) :pattern ((UOrdering3 alpha@@1py0 x@@7py0 y@@0py0) (UOrdering3 alpha@@1py0 y@@0py0 x@@7py0)) )) :named A4))
(assert (! (= (Ctor NormalFieldType) 6) :named A5_0))
(assert (! (= (Ctor RefType) 7) :named A5_1))
(assert (! (forall ((t0py0 T@T)(t1py0 T@T)(t2py0 T@T)(valpy0 T@U)(mpy0 T@U)(x0py0 T@U)(x1py0 T@U)) (! (= (MapType0Select t0py0 t1py0 t2py0 (MapType0Store t0py0 t1py0 t2py0 mpy0 x0py0 x1py0 valpy0) x0py0 x1py0) valpy0) :pattern ((MapType0Store t0py0 t1py0 t2py0 mpy0 x0py0 x1py0 valpy0)) )) :named A5_2))
(assert (! (forall ((u0py0 T@T)(s0py0 T@T)(s1py0 T@T)(t0@@0py0 T@T)(t1@@0py0 T@T)(val@@0py0 T@U)(m@@0py0 T@U)(x0@@0py0 T@U)(x1@@0py0 T@U)(y0py0 T@U)(y1py0 T@U)) (! (or (= s0py0 t0@@0py0) (= (MapType0Select t0@@0py0 t1@@0py0 u0py0 (MapType0Store s0py0 s1py0 u0py0 m@@0py0 x0@@0py0 x1@@0py0 val@@0py0) y0py0 y1py0) (MapType0Select t0@@0py0 t1@@0py0 u0py0 m@@0py0 y0py0 y1py0))) :pattern ((MapType0Select t0@@0py0 t1@@0py0 u0py0 (MapType0Store s0py0 s1py0 u0py0 m@@0py0 x0@@0py0 x1@@0py0 val@@0py0) y0py0 y1py0)) )) :named A5_3))
(assert (! (forall ((u0@@0py0 T@T)(s0@@0py0 T@T)(s1@@0py0 T@T)(t0@@1py0 T@T)(t1@@1py0 T@T)(val@@1py0 T@U)(m@@1py0 T@U)(x0@@1py0 T@U)(x1@@1py0 T@U)(y0@@0py0 T@U)(y1@@0py0 T@U)) (! (or (= s1@@0py0 t1@@1py0) (= (MapType0Select t0@@1py0 t1@@1py0 u0@@0py0 (MapType0Store s0@@0py0 s1@@0py0 u0@@0py0 m@@1py0 x0@@1py0 x1@@1py0 val@@1py0) y0@@0py0 y1@@0py0) (MapType0Select t0@@1py0 t1@@1py0 u0@@0py0 m@@1py0 y0@@0py0 y1@@0py0))) :pattern ((MapType0Select t0@@1py0 t1@@1py0 u0@@0py0 (MapType0Store s0@@0py0 s1@@0py0 u0@@0py0 m@@1py0 x0@@1py0 x1@@1py0 val@@1py0) y0@@0py0 y1@@0py0)) )) :named A5_4))
(assert (! (forall ((u0@@1py0 T@T)(s0@@1py0 T@T)(s1@@1py0 T@T)(t0@@2py0 T@T)(t1@@2py0 T@T)(val@@2py0 T@U)(m@@2py0 T@U)(x0@@2py0 T@U)(x1@@2py0 T@U)(y0@@1py0 T@U)(y1@@1py0 T@U)) (! (or (= x0@@2py0 y0@@1py0) (= (MapType0Select t0@@2py0 t1@@2py0 u0@@1py0 (MapType0Store s0@@1py0 s1@@1py0 u0@@1py0 m@@2py0 x0@@2py0 x1@@2py0 val@@2py0) y0@@1py0 y1@@1py0) (MapType0Select t0@@2py0 t1@@2py0 u0@@1py0 m@@2py0 y0@@1py0 y1@@1py0))) :pattern ((MapType0Select t0@@2py0 t1@@2py0 u0@@1py0 (MapType0Store s0@@1py0 s1@@1py0 u0@@1py0 m@@2py0 x0@@2py0 x1@@2py0 val@@2py0) y0@@1py0 y1@@1py0)) )) :named A5_5))
(assert (! (forall ((u0@@2py0 T@T)(s0@@2py0 T@T)(s1@@2py0 T@T)(t0@@3py0 T@T)(t1@@3py0 T@T)(val@@3py0 T@U)(m@@3py0 T@U)(x0@@3py0 T@U)(x1@@3py0 T@U)(y0@@2py0 T@U)(y1@@2py0 T@U)) (! (or (= x1@@3py0 y1@@2py0) (= (MapType0Select t0@@3py0 t1@@3py0 u0@@2py0 (MapType0Store s0@@2py0 s1@@2py0 u0@@2py0 m@@3py0 x0@@3py0 x1@@3py0 val@@3py0) y0@@2py0 y1@@2py0) (MapType0Select t0@@3py0 t1@@3py0 u0@@2py0 m@@3py0 y0@@2py0 y1@@2py0))) :pattern ((MapType0Select t0@@3py0 t1@@3py0 u0@@2py0 (MapType0Store s0@@2py0 s1@@2py0 u0@@2py0 m@@3py0 x0@@3py0 x1@@3py0 val@@3py0) y0@@2py0 y1@@2py0)) )) :named A5_6))
(assert (! (forall ((opy0 T@U)(fpy0 T@U)(Heappy0 T@U)) (! (or (U_2_bool (MapType0Select NormalFieldType boolType RefType Heappy0 (MapType0Select NormalFieldType RefType RefType Heappy0 opy0 fpy0) $allocated)) (not (U_2_bool (MapType0Select NormalFieldType boolType RefType Heappy0 opy0 $allocated)))) :pattern ((MapType0Select NormalFieldType RefType RefType Heappy0 opy0 fpy0)) )) :named A6))
(assert (! (forall ((Heap@@0py0 T@U)(ExhaleHeappy0 T@U)(Maskpy0 T@U)(o@@0py0 T@U)(f_2py0 T@U)(Apy0 T@T)(Bpy0 T@T)) (! (or (not (IdenticalOnKnownLocations Heap@@0py0 ExhaleHeappy0 Maskpy0)) (not (HasDirectPerm Apy0 Bpy0 Maskpy0 o@@0py0 f_2py0)) (= (MapType0Select Apy0 Bpy0 RefType Heap@@0py0 o@@0py0 f_2py0) (MapType0Select Apy0 Bpy0 RefType ExhaleHeappy0 o@@0py0 f_2py0))) :pattern ((IdenticalOnKnownLocations Heap@@0py0 ExhaleHeappy0 Maskpy0) (MapType0Select Apy0 Bpy0 RefType ExhaleHeappy0 o@@0py0 f_2py0)) )) :named A7))
(assert (! (= (Ctor FrameTypeType) 8) :named A8_0))
(assert (! (forall ((arg0@@5py0 T@T)(arg1py0 T@T)) (! (= (Ctor (MapType1Type arg0@@5py0 arg1py0)) 9) :pattern ((MapType1Type arg0@@5py0 arg1py0)) )) :named A8_1))
(assert (! (forall ((arg0@@6py0 T@T)(arg1@@0py0 T@T)) (! (= (MapType1TypeInv0 (MapType1Type arg0@@6py0 arg1@@0py0)) arg0@@6py0) :pattern ((MapType1Type arg0@@6py0 arg1@@0py0)) )) :named A8_2))
(assert (! (forall ((arg0@@7py0 T@T)(arg1@@1py0 T@T)) (! (= (MapType1TypeInv1 (MapType1Type arg0@@7py0 arg1@@1py0)) arg1@@1py0) :pattern ((MapType1Type arg0@@7py0 arg1@@1py0)) )) :named A8_3))
(assert (! (forall ((t0@@4py0 T@T)(t1@@4py0 T@T)(t2@@0py0 T@T)(t3py0 T@T)(val@@4py0 T@U)(m@@4py0 T@U)(x0@@4py0 T@U)(x1@@4py0 T@U)) (! (= (MapType1Select t0@@4py0 t1@@4py0 t2@@0py0 t3py0 (MapType1Store t0@@4py0 t1@@4py0 t2@@0py0 t3py0 m@@4py0 x0@@4py0 x1@@4py0 val@@4py0) x0@@4py0 x1@@4py0) val@@4py0) :pattern ((MapType1Store t0@@4py0 t1@@4py0 t2@@0py0 t3py0 m@@4py0 x0@@4py0 x1@@4py0 val@@4py0)) )) :named A8_4))
(assert (! (forall ((u0@@3py0 T@T)(u1py0 T@T)(s0@@3py0 T@T)(s1@@3py0 T@T)(t0@@5py0 T@T)(t1@@5py0 T@T)(val@@5py0 T@U)(m@@5py0 T@U)(x0@@5py0 T@U)(x1@@5py0 T@U)(y0@@3py0 T@U)(y1@@3py0 T@U)) (! (or (= s0@@3py0 t0@@5py0) (= (MapType1Select t0@@5py0 t1@@5py0 u0@@3py0 u1py0 (MapType1Store s0@@3py0 s1@@3py0 u0@@3py0 u1py0 m@@5py0 x0@@5py0 x1@@5py0 val@@5py0) y0@@3py0 y1@@3py0) (MapType1Select t0@@5py0 t1@@5py0 u0@@3py0 u1py0 m@@5py0 y0@@3py0 y1@@3py0))) :pattern ((MapType1Select t0@@5py0 t1@@5py0 u0@@3py0 u1py0 (MapType1Store s0@@3py0 s1@@3py0 u0@@3py0 u1py0 m@@5py0 x0@@5py0 x1@@5py0 val@@5py0) y0@@3py0 y1@@3py0)) )) :named A8_5))
(assert (! (forall ((u0@@4py0 T@T)(u1@@0py0 T@T)(s0@@4py0 T@T)(s1@@4py0 T@T)(t0@@6py0 T@T)(t1@@6py0 T@T)(val@@6py0 T@U)(m@@6py0 T@U)(x0@@6py0 T@U)(x1@@6py0 T@U)(y0@@4py0 T@U)(y1@@4py0 T@U)) (! (or (= s1@@4py0 t1@@6py0) (= (MapType1Select t0@@6py0 t1@@6py0 u0@@4py0 u1@@0py0 (MapType1Store s0@@4py0 s1@@4py0 u0@@4py0 u1@@0py0 m@@6py0 x0@@6py0 x1@@6py0 val@@6py0) y0@@4py0 y1@@4py0) (MapType1Select t0@@6py0 t1@@6py0 u0@@4py0 u1@@0py0 m@@6py0 y0@@4py0 y1@@4py0))) :pattern ((MapType1Select t0@@6py0 t1@@6py0 u0@@4py0 u1@@0py0 (MapType1Store s0@@4py0 s1@@4py0 u0@@4py0 u1@@0py0 m@@6py0 x0@@6py0 x1@@6py0 val@@6py0) y0@@4py0 y1@@4py0)) )) :named A8_6))
(assert (! (forall ((u0@@5py0 T@T)(u1@@1py0 T@T)(s0@@5py0 T@T)(s1@@5py0 T@T)(t0@@7py0 T@T)(t1@@7py0 T@T)(val@@7py0 T@U)(m@@7py0 T@U)(x0@@7py0 T@U)(x1@@7py0 T@U)(y0@@5py0 T@U)(y1@@5py0 T@U)) (! (or (= x0@@7py0 y0@@5py0) (= (MapType1Select t0@@7py0 t1@@7py0 u0@@5py0 u1@@1py0 (MapType1Store s0@@5py0 s1@@5py0 u0@@5py0 u1@@1py0 m@@7py0 x0@@7py0 x1@@7py0 val@@7py0) y0@@5py0 y1@@5py0) (MapType1Select t0@@7py0 t1@@7py0 u0@@5py0 u1@@1py0 m@@7py0 y0@@5py0 y1@@5py0))) :pattern ((MapType1Select t0@@7py0 t1@@7py0 u0@@5py0 u1@@1py0 (MapType1Store s0@@5py0 s1@@5py0 u0@@5py0 u1@@1py0 m@@7py0 x0@@7py0 x1@@7py0 val@@7py0) y0@@5py0 y1@@5py0)) )) :named A8_7))
(assert (! (forall ((u0@@6py0 T@T)(u1@@2py0 T@T)(s0@@6py0 T@T)(s1@@6py0 T@T)(t0@@8py0 T@T)(t1@@8py0 T@T)(val@@8py0 T@U)(m@@8py0 T@U)(x0@@8py0 T@U)(x1@@8py0 T@U)(y0@@6py0 T@U)(y1@@6py0 T@U)) (! (or (= x1@@8py0 y1@@6py0) (= (MapType1Select t0@@8py0 t1@@8py0 u0@@6py0 u1@@2py0 (MapType1Store s0@@6py0 s1@@6py0 u0@@6py0 u1@@2py0 m@@8py0 x0@@8py0 x1@@8py0 val@@8py0) y0@@6py0 y1@@6py0) (MapType1Select t0@@8py0 t1@@8py0 u0@@6py0 u1@@2py0 m@@8py0 y0@@6py0 y1@@6py0))) :pattern ((MapType1Select t0@@8py0 t1@@8py0 u0@@6py0 u1@@2py0 (MapType1Store s0@@6py0 s1@@6py0 u0@@6py0 u1@@2py0 m@@8py0 x0@@8py0 x1@@8py0 val@@8py0) y0@@6py0 y1@@6py0)) )) :named A8_8))
(assert (! (forall ((Heap@@1py0 T@U)(ExhaleHeap@@0py0 T@U)(Mask@@0py0 T@U)(pm_fpy0 T@U)(Cpy0 T@T)) (! (or (= (MapType0Select Cpy0 (MapType1Type RefType boolType) RefType Heap@@1py0 null (PredicateMaskField Cpy0 pm_fpy0)) (MapType0Select Cpy0 (MapType1Type RefType boolType) RefType ExhaleHeap@@0py0 null (PredicateMaskField Cpy0 pm_fpy0))) (not (IdenticalOnKnownLocations Heap@@1py0 ExhaleHeap@@0py0 Mask@@0py0)) (not (HasDirectPerm Cpy0 FrameTypeType Mask@@0py0 null pm_fpy0)) (not (IsPredicateField Cpy0 FrameTypeType pm_fpy0))) :pattern ((IdenticalOnKnownLocations Heap@@1py0 ExhaleHeap@@0py0 Mask@@0py0) (IsPredicateField Cpy0 FrameTypeType pm_fpy0) (MapType0Select Cpy0 (MapType1Type RefType boolType) RefType ExhaleHeap@@0py0 null (PredicateMaskField Cpy0 pm_fpy0))) )) :named A9))
(assert (! (forall ((Heap@@2py0 T@U)(ExhaleHeap@@1py0 T@U)(Mask@@1py0 T@U)(pm_f@@0py0 T@U)(C@@0py0 T@T)) (! (or (forall ((o2py0 T@U)(f_2@@0py0 T@U)(A@@0py0 T@T)(B@@0py0 T@T)) (! (or (not (U_2_bool (MapType1Select A@@0py0 B@@0py0 RefType boolType (MapType0Select C@@0py0 (MapType1Type RefType boolType) RefType Heap@@2py0 null (PredicateMaskField C@@0py0 pm_f@@0py0)) o2py0 f_2@@0py0))) (= (MapType0Select A@@0py0 B@@0py0 RefType Heap@@2py0 o2py0 f_2@@0py0) (MapType0Select A@@0py0 B@@0py0 RefType ExhaleHeap@@1py0 o2py0 f_2@@0py0))) :pattern ((MapType0Select A@@0py0 B@@0py0 RefType ExhaleHeap@@1py0 o2py0 f_2@@0py0)) )) (not (IdenticalOnKnownLocations Heap@@2py0 ExhaleHeap@@1py0 Mask@@1py0)) (not (HasDirectPerm C@@0py0 FrameTypeType Mask@@1py0 null pm_f@@0py0)) (not (IsPredicateField C@@0py0 FrameTypeType pm_f@@0py0))) :pattern ((IdenticalOnKnownLocations Heap@@2py0 ExhaleHeap@@1py0 Mask@@1py0) (MapType0Select C@@0py0 FrameTypeType RefType ExhaleHeap@@1py0 null pm_f@@0py0) (IsPredicateField C@@0py0 FrameTypeType pm_f@@0py0)) )) :named A10))
(assert (! (forall ((Heap@@3py0 T@U)(ExhaleHeap@@2py0 T@U)(Mask@@2py0 T@U)(pm_f@@1py0 T@U)(C@@1py0 T@T)) (! (or (= (MapType0Select C@@1py0 (MapType1Type RefType boolType) RefType Heap@@3py0 null (WandMaskField C@@1py0 pm_f@@1py0)) (MapType0Select C@@1py0 (MapType1Type RefType boolType) RefType ExhaleHeap@@2py0 null (WandMaskField C@@1py0 pm_f@@1py0))) (not (IdenticalOnKnownLocations Heap@@3py0 ExhaleHeap@@2py0 Mask@@2py0)) (not (HasDirectPerm C@@1py0 FrameTypeType Mask@@2py0 null pm_f@@1py0)) (not (IsWandField C@@1py0 FrameTypeType pm_f@@1py0))) :pattern ((IdenticalOnKnownLocations Heap@@3py0 ExhaleHeap@@2py0 Mask@@2py0) (IsWandField C@@1py0 FrameTypeType pm_f@@1py0) (MapType0Select C@@1py0 (MapType1Type RefType boolType) RefType ExhaleHeap@@2py0 null (WandMaskField C@@1py0 pm_f@@1py0))) )) :named A11))
(assert (! (forall ((Heap@@4py0 T@U)(ExhaleHeap@@3py0 T@U)(Mask@@3py0 T@U)(pm_f@@2py0 T@U)(C@@2py0 T@T)) (! (or (forall ((o2@@0py0 T@U)(f_2@@1py0 T@U)(A@@1py0 T@T)(B@@1py0 T@T)) (! (or (not (U_2_bool (MapType1Select A@@1py0 B@@1py0 RefType boolType (MapType0Select C@@2py0 (MapType1Type RefType boolType) RefType Heap@@4py0 null (WandMaskField C@@2py0 pm_f@@2py0)) o2@@0py0 f_2@@1py0))) (= (MapType0Select A@@1py0 B@@1py0 RefType Heap@@4py0 o2@@0py0 f_2@@1py0) (MapType0Select A@@1py0 B@@1py0 RefType ExhaleHeap@@3py0 o2@@0py0 f_2@@1py0))) :pattern ((MapType0Select A@@1py0 B@@1py0 RefType ExhaleHeap@@3py0 o2@@0py0 f_2@@1py0)) )) (not (IdenticalOnKnownLocations Heap@@4py0 ExhaleHeap@@3py0 Mask@@3py0)) (not (HasDirectPerm C@@2py0 FrameTypeType Mask@@3py0 null pm_f@@2py0)) (not (IsWandField C@@2py0 FrameTypeType pm_f@@2py0))) :pattern ((IdenticalOnKnownLocations Heap@@4py0 ExhaleHeap@@3py0 Mask@@3py0) (IsWandField C@@2py0 FrameTypeType pm_f@@2py0)) )) :named A12))
(assert (! (forall ((Heap@@5py0 T@U)(ExhaleHeap@@4py0 T@U)(Mask@@4py0 T@U)(o@@1py0 T@U)) (! (or (U_2_bool (MapType0Select NormalFieldType boolType RefType ExhaleHeap@@4py0 o@@1py0 $allocated)) (not (IdenticalOnKnownLocations Heap@@5py0 ExhaleHeap@@4py0 Mask@@4py0)) (not (U_2_bool (MapType0Select NormalFieldType boolType RefType Heap@@5py0 o@@1py0 $allocated)))) :pattern ((IdenticalOnKnownLocations Heap@@5py0 ExhaleHeap@@4py0 Mask@@4py0) (MapType0Select NormalFieldType boolType RefType ExhaleHeap@@4py0 o@@1py0 $allocated)) )) :named A13))
(assert (! (forall ((Heap@@6py0 T@U)(o@@2py0 T@U)(f_2@@2py0 T@U)(vpy0 T@U)(A@@2py0 T@T)(B@@2py0 T@T)) (! (succHeap Heap@@6py0 (MapType0Store A@@2py0 B@@2py0 RefType Heap@@6py0 o@@2py0 f_2@@2py0 vpy0)) :pattern ((MapType0Store A@@2py0 B@@2py0 RefType Heap@@6py0 o@@2py0 f_2@@2py0 vpy0)) )) :named A14))
(assert (! (forall ((Heap@@7py0 T@U)(ExhaleHeap@@5py0 T@U)(Mask@@5py0 T@U)) (! (or (not (IdenticalOnKnownLocations Heap@@7py0 ExhaleHeap@@5py0 Mask@@5py0)) (succHeap Heap@@7py0 ExhaleHeap@@5py0)) :pattern ((IdenticalOnKnownLocations Heap@@7py0 ExhaleHeap@@5py0 Mask@@5py0)) )) :named A15))
(assert (! (forall ((Heap0py0 T@U)(Heap1py0 T@U)) (! (or (not (succHeap Heap0py0 Heap1py0)) (succHeapTrans Heap0py0 Heap1py0)) :pattern ((succHeap Heap0py0 Heap1py0)) )) :named A16))
(assert (! (forall ((Heap0@@0py0 T@U)(Heap1@@0py0 T@U)(Heap2py0 T@U)) (! (or (succHeapTrans Heap0@@0py0 Heap2py0) (not (succHeapTrans Heap0@@0py0 Heap1@@0py0)) (not (succHeap Heap1@@0py0 Heap2py0))) :pattern ((succHeapTrans Heap0@@0py0 Heap1@@0py0) (succHeap Heap1@@0py0 Heap2py0)) )) :named A17))
(assert (! (forall ((o_1py0 T@U)(f_3py0 T@U)(A@@3py0 T@T)(B@@3py0 T@T)) (! (= (U_2_real (MapType1Select A@@3py0 B@@3py0 RefType realType ZeroMask o_1py0 f_3py0)) NoPerm) :pattern ((MapType1Select A@@3py0 B@@3py0 RefType realType ZeroMask o_1py0 f_3py0)) )) :named A18))
(assert (! (forall ((o_1@@0py0 T@U)(f_3@@0py0 T@U)(A@@4py0 T@T)(B@@4py0 T@T)) (! (not (U_2_bool (MapType1Select A@@4py0 B@@4py0 RefType boolType ZeroPMask o_1@@0py0 f_3@@0py0))) :pattern ((MapType1Select A@@4py0 B@@4py0 RefType boolType ZeroPMask o_1@@0py0 f_3@@0py0)) )) :named A19))
(assert (! (= NoPerm 0.0) :named A20))
(assert (! (= FullPerm 1.0) :named A21))
(assert (! (forall ((Heap@@8py0 T@U)(Mask@@6py0 T@U)) (! (or (GoodMask Mask@@6py0) (not (state Heap@@8py0 Mask@@6py0))) :pattern ((state Heap@@8py0 Mask@@6py0)) )) :named A22))
(assert (! (forall ((Mask@@7py0 T@U)(o_1@@1py0 T@U)(f_3@@1py0 T@U)(A@@5py0 T@T)(B@@5py0 T@T)) (! (or (not (GoodMask Mask@@7py0)) (and (<= NoPerm (U_2_real (MapType1Select A@@5py0 B@@5py0 RefType realType Mask@@7py0 o_1@@1py0 f_3@@1py0))) (or (IsPredicateField A@@5py0 B@@5py0 f_3@@1py0) (IsWandField A@@5py0 B@@5py0 f_3@@1py0) (<= (U_2_real (MapType1Select A@@5py0 B@@5py0 RefType realType Mask@@7py0 o_1@@1py0 f_3@@1py0)) FullPerm) (not (GoodMask Mask@@7py0))))) :pattern ((GoodMask Mask@@7py0) (MapType1Select A@@5py0 B@@5py0 RefType realType Mask@@7py0 o_1@@1py0 f_3@@1py0)) )) :named A23))
(assert (! (forall ((Mask@@8py0 T@U)(o_1@@2py0 T@U)(f_3@@2py0 T@U)(A@@6py0 T@T)(B@@6py0 T@T)) (! (and (or (not (HasDirectPerm A@@6py0 B@@6py0 Mask@@8py0 o_1@@2py0 f_3@@2py0)) (not (<= (U_2_real (MapType1Select A@@6py0 B@@6py0 RefType realType Mask@@8py0 o_1@@2py0 f_3@@2py0)) NoPerm))) (or (HasDirectPerm A@@6py0 B@@6py0 Mask@@8py0 o_1@@2py0 f_3@@2py0) (<= (U_2_real (MapType1Select A@@6py0 B@@6py0 RefType realType Mask@@8py0 o_1@@2py0 f_3@@2py0)) NoPerm))) :pattern ((HasDirectPerm A@@6py0 B@@6py0 Mask@@8py0 o_1@@2py0 f_3@@2py0)) )) :named A24))
(assert (! (forall ((ResultMaskpy0 T@U)(SummandMask1py0 T@U)(SummandMask2py0 T@U)(o_1@@3py0 T@U)(f_3@@3py0 T@U)(A@@7py0 T@T)(B@@7py0 T@T)) (! (or (not (sumMask ResultMaskpy0 SummandMask1py0 SummandMask2py0)) (= (U_2_real (MapType1Select A@@7py0 B@@7py0 RefType realType ResultMaskpy0 o_1@@3py0 f_3@@3py0)) (+ (U_2_real (MapType1Select A@@7py0 B@@7py0 RefType realType SummandMask1py0 o_1@@3py0 f_3@@3py0)) (U_2_real (MapType1Select A@@7py0 B@@7py0 RefType realType SummandMask2py0 o_1@@3py0 f_3@@3py0))))) :pattern ((sumMask ResultMaskpy0 SummandMask1py0 SummandMask2py0) (MapType1Select A@@7py0 B@@7py0 RefType realType ResultMaskpy0 o_1@@3py0 f_3@@3py0)) :pattern ((sumMask ResultMaskpy0 SummandMask1py0 SummandMask2py0) (MapType1Select A@@7py0 B@@7py0 RefType realType SummandMask1py0 o_1@@3py0 f_3@@3py0)) :pattern ((sumMask ResultMaskpy0 SummandMask1py0 SummandMask2py0) (MapType1Select A@@7py0 B@@7py0 RefType realType SummandMask2py0 o_1@@3py0 f_3@@3py0)) )) :named A25))
(assert (! (forall ((ppy0 Real)(f_5py0 T@U)) (! (= (ConditionalFrame ppy0 f_5py0) (z3name!0 f_5py0 ppy0)) :pattern ((ConditionalFrame ppy0 f_5py0)) )) :named A26_0))
(assert (! (forall ((x!1py0 Real)(x!2py0 T@U)) (! (or (<= x!1py0 0.0) (= (z3name!0 x!2py0 x!1py0) x!2py0)) :pattern ((z3name!0 x!2py0 x!1py0)) )) :named A26_1))
(assert (! (forall ((x!1py1 Real)(x!2py1 T@U)) (! (or (not (<= x!1py1 0.0)) (= (z3name!0 x!2py1 x!1py1) EmptyFrame)) :pattern ((z3name!0 x!2py1 x!1py1)) )) :named A26_2))
(assert (! (forall ((p@@0py0 T@U)(v_1py0 T@U)(qpy0 T@U)(wpy0 T@U)(rpy0 T@U)(upy0 T@U)(A@@8py0 T@T)(B@@8py0 T@T)(C@@3py0 T@T)) (! (or (InsidePredicate A@@8py0 C@@3py0 p@@0py0 v_1py0 rpy0 upy0) (not (InsidePredicate A@@8py0 B@@8py0 p@@0py0 v_1py0 qpy0 wpy0)) (not (InsidePredicate B@@8py0 C@@3py0 qpy0 wpy0 rpy0 upy0))) :pattern ((InsidePredicate A@@8py0 B@@8py0 p@@0py0 v_1py0 qpy0 wpy0) (InsidePredicate B@@8py0 C@@3py0 qpy0 wpy0 rpy0 upy0)) )) :named A27))
(assert (! (forall ((p@@1py0 T@U)(v_1@@0py0 T@U)(w@@0py0 T@U)(A@@9py0 T@T)) (! (not (InsidePredicate A@@9py0 A@@9py0 p@@1py0 v_1@@0py0 p@@1py0 w@@0py0)) :pattern ((InsidePredicate A@@9py0 A@@9py0 p@@1py0 v_1@@0py0 p@@1py0 w@@0py0)) )) :named A28))
(assert (! (forall ((Heap@@9py0 T@U)(this$_2py0 T@U)(x@@8py0 Int)(y@@1py0 Int)(z@@0py0 Int)) (! (and (or (not (Triggersf$ Heap@@9py0 this$_2py0 x@@8py0 y@@1py0 z@@0py0)) (Triggersf$@quote@ Heap@@9py0 this$_2py0 x@@8py0 y@@1py0 z@@0py0)) (or (Triggersf$ Heap@@9py0 this$_2py0 x@@8py0 y@@1py0 z@@0py0) (not (Triggersf$@quote@ Heap@@9py0 this$_2py0 x@@8py0 y@@1py0 z@@0py0))) (dummyFunction boolType (bool_2_U (Triggersf$@sharp@triggerStateless this$_2py0 x@@8py0 y@@1py0 z@@0py0)))) :pattern ((Triggersf$ Heap@@9py0 this$_2py0 x@@8py0 y@@1py0 z@@0py0)) )) :named A29))
(assert (! (forall ((Heap@@10py0 T@U)(this$_2@@0py0 T@U)(x@@9py0 Int)(y@@2py0 Int)(z@@1py0 Int)) (! (dummyFunction boolType (bool_2_U (Triggersf$@sharp@triggerStateless this$_2@@0py0 x@@9py0 y@@2py0 z@@1py0))) :pattern ((Triggersf$@quote@ Heap@@10py0 this$_2@@0py0 x@@9py0 y@@2py0 z@@1py0)) )) :named A30))
(assert (! (= (Ctor PredicateType_Triggersvalid$Type) 10) :named A31))
(assert (! (forall ((Heap@@11py0 T@U)(Mask@@9py0 T@U)(this$_2@@1py0 T@U)(x@@10py0 Int)(y@@3py0 Int)(z@@2py0 Int)) (! (or (not (state Heap@@11py0 Mask@@9py0)) (and (or (not (Triggersf$@quote@ Heap@@11py0 this$_2@@1py0 x@@10py0 y@@3py0 z@@2py0)) (Triggersf$@sharp@frame (MapType0Select PredicateType_Triggersvalid$Type FrameTypeType RefType Heap@@11py0 null (Triggersvalid$ this$_2@@1py0)) this$_2@@1py0 x@@10py0 y@@3py0 z@@2py0)) (or (Triggersf$@quote@ Heap@@11py0 this$_2@@1py0 x@@10py0 y@@3py0 z@@2py0) (not (Triggersf$@sharp@frame (MapType0Select PredicateType_Triggersvalid$Type FrameTypeType RefType Heap@@11py0 null (Triggersvalid$ this$_2@@1py0)) this$_2@@1py0 x@@10py0 y@@3py0 z@@2py0))))) :pattern ((state Heap@@11py0 Mask@@9py0) (Triggersf$@quote@ Heap@@11py0 this$_2@@1py0 x@@10py0 y@@3py0 z@@2py0)) )) :named A32))
(assert (! (forall ((Heap@@12py0 T@U)(this$_4py0 T@U)(x@@11py0 Int)) (! (and (or (not (Triggersg$ Heap@@12py0 this$_4py0 x@@11py0)) (Triggersg$@quote@ Heap@@12py0 this$_4py0 x@@11py0)) (or (Triggersg$ Heap@@12py0 this$_4py0 x@@11py0) (not (Triggersg$@quote@ Heap@@12py0 this$_4py0 x@@11py0))) (dummyFunction boolType (bool_2_U (Triggersg$@sharp@triggerStateless this$_4py0 x@@11py0)))) :pattern ((Triggersg$ Heap@@12py0 this$_4py0 x@@11py0)) )) :named A33))
(assert (! (forall ((Heap@@13py0 T@U)(this$_4@@0py0 T@U)(x@@12py0 Int)) (! (dummyFunction boolType (bool_2_U (Triggersg$@sharp@triggerStateless this$_4@@0py0 x@@12py0))) :pattern ((Triggersg$@quote@ Heap@@13py0 this$_4@@0py0 x@@12py0)) )) :named A34))
(assert (! (forall ((Heap@@14py0 T@U)(Mask@@10py0 T@U)(this$_4@@1py0 T@U)(x@@13py0 Int)) (! (or (not (state Heap@@14py0 Mask@@10py0)) (and (or (not (Triggersg$@quote@ Heap@@14py0 this$_4@@1py0 x@@13py0)) (Triggersg$@sharp@frame (MapType0Select PredicateType_Triggersvalid$Type FrameTypeType RefType Heap@@14py0 null (Triggersvalid$ this$_4@@1py0)) this$_4@@1py0 x@@13py0)) (or (Triggersg$@quote@ Heap@@14py0 this$_4@@1py0 x@@13py0) (not (Triggersg$@sharp@frame (MapType0Select PredicateType_Triggersvalid$Type FrameTypeType RefType Heap@@14py0 null (Triggersvalid$ this$_4@@1py0)) this$_4@@1py0 x@@13py0))))) :pattern ((state Heap@@14py0 Mask@@10py0) (Triggersg$@quote@ Heap@@14py0 this$_4@@1py0 x@@13py0)) )) :named A35))
(assert (! (forall ((this$_1py0 T@U)) (! (= (PredicateMaskField PredicateType_Triggersvalid$Type (Triggersvalid$ this$_1py0)) (Triggersvalid$@sharp@sm this$_1py0)) :pattern ((PredicateMaskField PredicateType_Triggersvalid$Type (Triggersvalid$ this$_1py0))) )) :named A36))
(assert (! (forall ((this$_1@@0py0 T@U)) (! (IsPredicateField PredicateType_Triggersvalid$Type FrameTypeType (Triggersvalid$ this$_1@@0py0)) :pattern ((Triggersvalid$ this$_1@@0py0)) )) :named A37))
(assert (! (forall ((this$_1@@1py0 T@U)) (! (= (getPredicateId PredicateType_Triggersvalid$Type FrameTypeType (Triggersvalid$ this$_1@@1py0)) 0) :pattern ((Triggersvalid$ this$_1@@1py0)) )) :named A38))
(assert (! (forall ((this$_1@@2py0 T@U)(this$_12py0 T@U)) (! (or (not (= (Triggersvalid$ this$_1@@2py0) (Triggersvalid$ this$_12py0))) (= this$_1@@2py0 this$_12py0)) :pattern ((Triggersvalid$ this$_1@@2py0) (Triggersvalid$ this$_12py0)) )) :named A39))
(assert (! (forall ((this$_1@@3py0 T@U)(this$_12@@0py0 T@U)) (! (or (not (= (Triggersvalid$@sharp@sm this$_1@@3py0) (Triggersvalid$@sharp@sm this$_12@@0py0))) (= this$_1@@3py0 this$_12@@0py0)) :pattern ((Triggersvalid$@sharp@sm this$_1@@3py0) (Triggersvalid$@sharp@sm this$_12@@0py0)) )) :named A40))
(assert (! (forall ((Heap@@15py0 T@U)(this$_1@@4py0 T@U)) (! (Triggersvalid$@sharp@everUsed PredicateType_Triggersvalid$Type (Triggersvalid$ this$_1@@4py0)) :pattern ((Triggersvalid$@sharp@trigger PredicateType_Triggersvalid$Type Heap@@15py0 (Triggersvalid$ this$_1@@4py0))) )) :named A41))
(assert (! (= (ControlFlow 0 0) 5148) :named A42_0))
(assert (! (= (ControlFlow 0 5148) 2731) :named A42_1))
(assert (! (= AssumeFunctionsAbove (- 1)) :named A42_2))
(assert (! (= Mask@0 (MapType1Store PredicateType_Triggersvalid$Type FrameTypeType RefType realType ZeroMask null (Triggersvalid$ this$_8) (real_2_U (+ (U_2_real (MapType1Select PredicateType_Triggersvalid$Type FrameTypeType RefType realType ZeroMask null (Triggersvalid$ this$_8))) FullPerm)))) :named A42_3))
(assert (! (U_2_bool (MapType0Select NormalFieldType boolType RefType Heap@@16 this$_8 $allocated)) :named A42_4))
(assert (! (not (<= FullPerm k$_2)) :named A42_5))
(assert (! (not (<= k$_2 NoPerm)) :named A42_6))
(assert (! (not (<= wildcard NoPerm)) :named A42_7))
(assert (! (not (= this$_8 null)) :named A42_8))
(assert (! (or (and (= (ControlFlow 0 2731) 2764) (state Heap@@16 Mask@0) (forall ((a_1_1py0 Int)) (! (and (Triggersg$ Heap@@16 this$_8 a_1_1py0) (forall ((b_1_1py0 Int)(c_1py0 Int)) (! (Triggersf$ Heap@@16 this$_8 a_1_1py0 b_1_1py0 c_1py0) :pattern ((Triggersf$@sharp@frame (MapType0Select PredicateType_Triggersvalid$Type FrameTypeType RefType Heap@@16 null (Triggersvalid$ this$_8)) this$_8 a_1_1py0 b_1_1py0 c_1py0)) ))) :pattern ((Triggersg$@sharp@frame (MapType0Select PredicateType_Triggersvalid$Type FrameTypeType RefType Heap@@16 null (Triggersvalid$ this$_8)) this$_8 a_1_1py0)) )) (= Mask@0 Mask@@11) (state PostHeap@0 ZeroMask) (or (and (= (ControlFlow 0 2764) 2768) (= PostMask@0 (MapType1Store PredicateType_Triggersvalid$Type FrameTypeType RefType realType ZeroMask null (Triggersvalid$ this$_8) (real_2_U (+ (U_2_real (MapType1Select PredicateType_Triggersvalid$Type FrameTypeType RefType realType ZeroMask null (Triggersvalid$ this$_8))) FullPerm)))) (state PostHeap@0 PostMask@0) (or (and (= (ControlFlow 0 2768) 2778) (state PostHeap@0 PostMask@0) (forall ((x_1py0 Int)(y_1py0 Int)(z_1py0 Int)) (! (Triggersf$ PostHeap@0 this$_8 x_1py0 y_1py0 z_1py0) :pattern ((Triggersf$@sharp@frame (MapType0Select PredicateType_Triggersvalid$Type FrameTypeType RefType PostHeap@0 null (Triggersvalid$ this$_8)) this$_8 x_1py0 y_1py0 z_1py0)) )) (= (ControlFlow 0 2778) 2784) (or (and (= (ControlFlow 0 2784) (- 5923)) (= this$_8 null)) (and (not (= this$_8 null)) (not (<= wildcard@0 NoPerm)) (= perm@1 (+ NoPerm wildcard@0)) (= (ControlFlow 0 2784) (- 5950)) (<= (U_2_real (MapType1Select PredicateType_Triggersvalid$Type FrameTypeType RefType realType PostMask@0 null (Triggersvalid$ this$_8))) NoPerm)))) (and (= (ControlFlow 0 2768) 2772) (or (and (= (ControlFlow 0 2772) (- 6017)) (= this$_8 null)) (and (not (= this$_8 null)) (not (<= wildcard@1 NoPerm)) (= perm@2 (+ NoPerm wildcard@1)) (= (ControlFlow 0 2772) (- 6044)) (<= (U_2_real (MapType1Select PredicateType_Triggersvalid$Type FrameTypeType RefType realType PostMask@0 null (Triggersvalid$ this$_8))) NoPerm)))))) (and (= (ControlFlow 0 2764) 2794) (U_2_bool (MapType0Select NormalFieldType boolType RefType Heap@@16 n$_2 $allocated)) (= perm@0 (+ NoPerm FullPerm)) (or (and (= (ControlFlow 0 2794) 2803) (not (= perm@0 NoPerm)) (or (and (= (ControlFlow 0 2803) (- 5693)) (not (<= perm@0 (U_2_real (MapType1Select PredicateType_Triggersvalid$Type FrameTypeType RefType realType Mask@0 null (Triggersvalid$ this$_8)))))) (and (<= perm@0 (U_2_real (MapType1Select PredicateType_Triggersvalid$Type FrameTypeType RefType realType Mask@0 null (Triggersvalid$ this$_8)))) (= (ControlFlow 0 2803) 2807) (= Mask@1 (MapType1Store PredicateType_Triggersvalid$Type FrameTypeType RefType realType Mask@0 null (Triggersvalid$ this$_8) (real_2_U (+ (U_2_real (MapType1Select PredicateType_Triggersvalid$Type FrameTypeType RefType realType Mask@0 null (Triggersvalid$ this$_8))) (* (- 1.0) perm@0))))) (or (and (= (ControlFlow 0 2807) 2809) (= (ControlFlow 0 2809) (- 5738)) (not (Triggersf$ Heap@@16 this$_8 x_2 y_2 z_2))) (and (= (ControlFlow 0 2807) 2811) (forall ((x_3_1py0 Int)(y_3_1py0 Int)(z_3_1py0 Int)) (! (Triggersf$ Heap@@16 this$_8 x_3_1py0 y_3_1py0 z_3_1py0) :pattern ((Triggersf$@sharp@frame (MapType0Select PredicateType_Triggersvalid$Type FrameTypeType RefType Heap@@16 null (Triggersvalid$ this$_8)) this$_8 x_3_1py0 y_3_1py0 z_3_1py0)) )) (= (ControlFlow 0 2811) 2815) (= (ControlFlow 0 2815) (- 5788)) (not (Triggersg$ Heap@@16 this$_8 w_2))))))) (and (= (ControlFlow 0 2794) 2805) (= perm@0 NoPerm) (= (ControlFlow 0 2805) 2807) (= Mask@1 (MapType1Store PredicateType_Triggersvalid$Type FrameTypeType RefType realType Mask@0 null (Triggersvalid$ this$_8) (real_2_U (+ (U_2_real (MapType1Select PredicateType_Triggersvalid$Type FrameTypeType RefType realType Mask@0 null (Triggersvalid$ this$_8))) (* (- 1.0) perm@0))))) (or (and (= (ControlFlow 0 2807) 2809) (= (ControlFlow 0 2809) (- 5738)) (not (Triggersf$ Heap@@16 this$_8 x_2 y_2 z_2))) (and (= (ControlFlow 0 2807) 2811) (forall ((x_3_1py1 Int)(y_3_1py1 Int)(z_3_1py1 Int)) (! (Triggersf$ Heap@@16 this$_8 x_3_1py1 y_3_1py1 z_3_1py1) :pattern ((Triggersf$@sharp@frame (MapType0Select PredicateType_Triggersvalid$Type FrameTypeType RefType Heap@@16 null (Triggersvalid$ this$_8)) this$_8 x_3_1py1 y_3_1py1 z_3_1py1)) )) (= (ControlFlow 0 2811) 2815) (= (ControlFlow 0 2815) (- 5788)) (not (Triggersg$ Heap@@16 this$_8 w_2))))))))) (and (= (ControlFlow 0 2731) 2735) (or (and (= (ControlFlow 0 2735) (- 6111)) (= this$_8 null)) (and (not (= this$_8 null)) (not (<= wildcard@2 NoPerm)) (= perm@3 (+ NoPerm wildcard@2)) (= (ControlFlow 0 2735) (- 6138)) (<= (U_2_real (MapType1Select PredicateType_Triggersvalid$Type FrameTypeType RefType realType Mask@0 null (Triggersvalid$ this$_8))) NoPerm)))) (and (= (ControlFlow 0 2731) 2748) (Triggersg$ Heap@@16 this$_8 a_2) (= (ControlFlow 0 2748) 2752) (or (and (= (ControlFlow 0 2752) (- 6224)) (= this$_8 null)) (and (not (= this$_8 null)) (not (<= wildcard@3 NoPerm)) (= perm@4 (+ NoPerm wildcard@3)) (= (ControlFlow 0 2752) (- 6251)) (<= (U_2_real (MapType1Select PredicateType_Triggersvalid$Type FrameTypeType RefType realType Mask@0 null (Triggersvalid$ this$_8))) NoPerm))))) :named A42_9))
(assert (! (state Heap@@16 Mask@0) :named A42_10))
(assert (! (state Heap@@16 ZeroMask) :named A42_11))
(check-sat)
(get-info :reason-unknown)
;z3 -T:600 smt-inputs/benchmarks/verifiers/viper/pattern_augmenter/ematching/tmp/b10_2.vpr-typeEncoding-a_std_unique_aug-gt_unsat-full.smt2
;unknown
;((:reason-unknown "smt tactic failed to show goal to be sat/unsat (incomplete quantifiers)"))
