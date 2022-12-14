(set-info :smt-lib-version 2.0)
(set-info :category "industrial")
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
(declare-fun int_2_U (Int) T@U)
(declare-fun U_2_int (T@U) Int)
(declare-fun type (T@U) T@T)
(declare-fun real_2_U (Real) T@U)
(declare-fun U_2_real (T@U) Real)
(declare-fun bool_2_U (Bool) T@U)
(declare-fun U_2_bool (T@U) Bool)
(declare-fun rmode_2_U (RMode) T@U)
(declare-fun U_2_rmode (T@U) RMode)
(declare-fun TyType () T@T)
(declare-fun TBool () T@U)
(declare-fun TChar () T@U)
(declare-fun TInt () T@U)
(declare-fun TReal () T@U)
(declare-fun TyTagType () T@T)
(declare-fun TagBool () T@U)
(declare-fun TagChar () T@U)
(declare-fun TagInt () T@U)
(declare-fun TagReal () T@U)
(declare-fun TagSet () T@U)
(declare-fun TagISet () T@U)
(declare-fun TagMultiSet () T@U)
(declare-fun TagSeq () T@U)
(declare-fun TagMap () T@U)
(declare-fun TagIMap () T@U)
(declare-fun TagClass () T@U)
(declare-fun ClassNameType () T@T)
(declare-fun NoTraitAtAll () T@U)
(declare-fun class._System.int () T@U)
(declare-fun class._System.bool () T@U)
(declare-fun class._System.set () T@U)
(declare-fun class._System.seq () T@U)
(declare-fun class._System.multiset () T@U)
(declare-fun FieldType (T@T) T@T)
(declare-fun FieldTypeInv0 (T@T) T@T)
(declare-fun alloc () T@U)
(declare-fun Tagclass._System.nat () T@U)
(declare-fun class._System.object () T@U)
(declare-fun Tagclass._System.object () T@U)
(declare-fun class._System.array () T@U)
(declare-fun Tagclass._System.array () T@U)
(declare-fun Tagclass._System.___hFunc0 () T@U)
(declare-fun Tagclass._System.___hPartialFunc0 () T@U)
(declare-fun Tagclass._System.___hTotalFunc0 () T@U)
(declare-fun class._System.__tuple_h0 () T@U)
(declare-fun DtCtorIdType () T@T)
(declare-fun |@sharp@@sharp@_System._tuple@sharp@0._@sharp@Make0| () T@U)
(declare-fun Tagclass._System.__tuple_h0 () T@U)
(declare-fun class._System.__tuple_h2 () T@U)
(declare-fun |@sharp@@sharp@_System._tuple@sharp@2._@sharp@Make2| () T@U)
(declare-fun Tagclass._System.__tuple_h2 () T@U)
(declare-fun Tagclass._System.___hFunc1 () T@U)
(declare-fun Tagclass._System.___hPartialFunc1 () T@U)
(declare-fun Tagclass._System.___hTotalFunc1 () T@U)
(declare-fun class._module.__default () T@U)
(declare-fun Tagclass._module.__default () T@U)
(declare-fun $$Language$Dafny () Bool)
(declare-fun TSet (T@U) T@U)
(declare-fun Inv0_TSet (T@U) T@U)
(declare-fun TISet (T@U) T@U)
(declare-fun Inv0_TISet (T@U) T@U)
(declare-fun TSeq (T@U) T@U)
(declare-fun Inv0_TSeq (T@U) T@U)
(declare-fun TMultiSet (T@U) T@U)
(declare-fun Inv0_TMultiSet (T@U) T@U)
(declare-fun TMap (T@U T@U) T@U)
(declare-fun Inv0_TMap (T@U) T@U)
(declare-fun Inv1_TMap (T@U) T@U)
(declare-fun TIMap (T@U T@U) T@U)
(declare-fun Inv0_TIMap (T@U) T@U)
(declare-fun Inv1_TIMap (T@U) T@U)
(declare-fun Tag (T@U) T@U)
(declare-fun LitInt (Int) Int)
(declare-fun BoxType () T@T)
(declare-fun $Box (T@U) T@U)
(declare-fun Lit (T@U) T@U)
(declare-fun LitReal (Real) Real)
(declare-fun charType () T@T)
(declare-fun |char@sharp@FromInt| (Int) T@U)
(declare-fun |char@sharp@ToInt| (T@U) Int)
(declare-fun $Unbox (T@T T@U) T@U)
(declare-fun $IsBox (T@U T@U) Bool)
(declare-fun $Is (T@U T@U) Bool)
(declare-fun MapType0Type (T@T T@T) T@T)
(declare-fun MapType0TypeInv0 (T@T) T@T)
(declare-fun MapType0TypeInv1 (T@T) T@T)
(declare-fun MapType0Select (T@U T@U) T@U)
(declare-fun MapType0Store (T@U T@U T@U) T@U)
(declare-fun SeqType (T@T) T@T)
(declare-fun SeqTypeInv0 (T@T) T@T)
(declare-fun MapType (T@T T@T) T@T)
(declare-fun MapTypeInv0 (T@T) T@T)
(declare-fun MapTypeInv1 (T@T) T@T)
(declare-fun IMapType (T@T T@T) T@T)
(declare-fun IMapTypeInv0 (T@T) T@T)
(declare-fun IMapTypeInv1 (T@T) T@T)
(declare-fun MapType1Type (T@T) T@T)
(declare-fun MapType1TypeInv0 (T@T) T@T)
(declare-fun MapType1Select (T@U T@U T@U) T@U)
(declare-fun MapType1Store (T@U T@U T@U T@U) T@U)
(declare-fun refType () T@T)
(declare-fun $IsAllocBox (T@U T@U T@U) Bool)
(declare-fun $IsAlloc (T@U T@U T@U) Bool)
(declare-fun $IsGoodMultiSet (T@U) Bool)
(declare-fun |Seq@sharp@Index| (T@U Int) T@U)
(declare-fun |Seq@sharp@Length| (T@U) Int)
(declare-fun |Map@sharp@Elements| (T@U) T@U)
(declare-fun |Map@sharp@Domain| (T@U) T@U)
(declare-fun |IMap@sharp@Elements| (T@U) T@U)
(declare-fun |IMap@sharp@Domain| (T@U) T@U)
(declare-fun TypeTuple (T@U T@U) T@U)
(declare-fun TypeTupleCar (T@U) T@U)
(declare-fun TypeTupleCdr (T@U) T@U)
(declare-fun SetRef_to_SetBox (T@U) T@U)
(declare-fun Tclass._System.object () T@U)
(declare-fun DatatypeTypeType () T@T)
(declare-fun BoxRank (T@U) Int)
(declare-fun DtRank (T@U) Int)
(declare-fun LayerTypeType () T@T)
(declare-fun AtLayer (T@U T@U) T@U)
(declare-fun $LS (T@U) T@U)
(declare-fun IndexField (Int) T@U)
(declare-fun FDim (T@U) Int)
(declare-fun IndexField_Inverse (T@U) Int)
(declare-fun MultiIndexField (T@U Int) T@U)
(declare-fun MultiIndexField_Inverse0 (T@U) T@U)
(declare-fun MultiIndexField_Inverse1 (T@U) Int)
(declare-fun NameFamilyType () T@T)
(declare-fun FieldOfDecl (T@T T@U T@U) T@U)
(declare-fun DeclType (T@U) T@U)
(declare-fun DeclName (T@U) T@U)
(declare-fun $HeapSucc (T@U T@U) Bool)
(declare-fun $IsGhostField (T@U) Bool)
(declare-fun _System.array.Length (T@U) Int)
(declare-fun q@Int (Real) Int)
(declare-fun q@Real (Int) Real)
(declare-fun $IsGoodHeap (T@U) Bool)
(declare-fun $HeapSuccGhost (T@U T@U) Bool)
(declare-fun |Set@sharp@Card| (T@U) Int)
(declare-fun |Set@sharp@Empty| (T@T) T@U)
(declare-fun |Set@sharp@Singleton| (T@U) T@U)
(declare-fun |Set@sharp@UnionOne| (T@U T@U) T@U)
(declare-fun |Set@sharp@Union| (T@U T@U) T@U)
(declare-fun |Set@sharp@Difference| (T@U T@U) T@U)
(declare-fun |Set@sharp@Disjoint| (T@U T@U) Bool)
(declare-fun |Set@sharp@Intersection| (T@U T@U) T@U)
(declare-fun |Set@sharp@Subset| (T@U T@U) Bool)
(declare-fun |Set@sharp@Equal| (T@U T@U) Bool)
(declare-fun |ISet@sharp@Empty| (T@T) T@U)
(declare-fun |ISet@sharp@UnionOne| (T@U T@U) T@U)
(declare-fun |ISet@sharp@Union| (T@U T@U) T@U)
(declare-fun |ISet@sharp@Difference| (T@U T@U) T@U)
(declare-fun |ISet@sharp@Disjoint| (T@U T@U) Bool)
(declare-fun |ISet@sharp@Intersection| (T@U T@U) T@U)
(declare-fun |ISet@sharp@Subset| (T@U T@U) Bool)
(declare-fun |ISet@sharp@Equal| (T@U T@U) Bool)
(declare-fun |Math@sharp@min| (Int Int) Int)
(declare-fun |Math@sharp@clip| (Int) Int)
(declare-fun |MultiSet@sharp@Card| (T@U) Int)
(declare-fun |MultiSet@sharp@Empty| (T@T) T@U)
(declare-fun |MultiSet@sharp@Singleton| (T@U) T@U)
(declare-fun |MultiSet@sharp@UnionOne| (T@U T@U) T@U)
(declare-fun |MultiSet@sharp@Union| (T@U T@U) T@U)
(declare-fun |MultiSet@sharp@Intersection| (T@U T@U) T@U)
(declare-fun |MultiSet@sharp@Difference| (T@U T@U) T@U)
(declare-fun |MultiSet@sharp@Subset| (T@U T@U) Bool)
(declare-fun |MultiSet@sharp@Equal| (T@U T@U) Bool)
(declare-fun |MultiSet@sharp@Disjoint| (T@U T@U) Bool)
(declare-fun |MultiSet@sharp@FromSet| (T@U) T@U)
(declare-fun |MultiSet@sharp@FromSeq| (T@U) T@U)
(declare-fun |Seq@sharp@Build| (T@U T@U) T@U)
(declare-fun |Seq@sharp@Empty| (T@T) T@U)
(declare-fun |Seq@sharp@Append| (T@U T@U) T@U)
(declare-fun |Seq@sharp@Update| (T@U Int T@U) T@U)
(declare-fun |Seq@sharp@Singleton| (T@U) T@U)
(declare-fun |Seq@sharp@Build_inv0| (T@U) T@U)
(declare-fun |Seq@sharp@Build_inv1| (T@U) T@U)
(declare-fun |Seq@sharp@Contains| (T@U T@U) Bool)
(declare-fun |Seq@sharp@Take| (T@U Int) T@U)
(declare-fun |Seq@sharp@Drop| (T@U Int) T@U)
(declare-fun |Seq@sharp@Equal| (T@U T@U) Bool)
(declare-fun |Seq@sharp@SameUntil| (T@U T@U Int) Bool)
(declare-fun |Seq@sharp@FromArray| (T@U T@U) T@U)
(declare-fun |Seq@sharp@Rank| (T@U) Int)
(declare-fun |Map@sharp@Card| (T@U) Int)
(declare-fun |Map@sharp@Values| (T@U) T@U)
(declare-fun |Map@sharp@Items| (T@U) T@U)
(declare-fun _System.__tuple_h2._0 (T@U) T@U)
(declare-fun _System.__tuple_h2._1 (T@U) T@U)
(declare-fun |Map@sharp@Empty| (T@T T@T) T@U)
(declare-fun |Map@sharp@Glue| (T@U T@U T@U) T@U)
(declare-fun |Map@sharp@Build| (T@U T@U T@U) T@U)
(declare-fun |Map@sharp@Equal| (T@U T@U) Bool)
(declare-fun |Map@sharp@Disjoint| (T@U T@U) Bool)
(declare-fun |IMap@sharp@Values| (T@U) T@U)
(declare-fun |IMap@sharp@Items| (T@U) T@U)
(declare-fun |IMap@sharp@Empty| (T@T T@T) T@U)
(declare-fun |IMap@sharp@Glue| (T@U T@U T@U) T@U)
(declare-fun |IMap@sharp@Build| (T@U T@U T@U) T@U)
(declare-fun |IMap@sharp@Equal| (T@U T@U) Bool)
(declare-fun INTERNAL_add_boogie (Int Int) Int)
(declare-fun INTERNAL_sub_boogie (Int Int) Int)
(declare-fun INTERNAL_mul_boogie (Int Int) Int)
(declare-fun INTERNAL_div_boogie (Int Int) Int)
(declare-fun INTERNAL_mod_boogie (Int Int) Int)
(declare-fun INTERNAL_lt_boogie (Int Int) Bool)
(declare-fun INTERNAL_le_boogie (Int Int) Bool)
(declare-fun INTERNAL_gt_boogie (Int Int) Bool)
(declare-fun INTERNAL_ge_boogie (Int Int) Bool)
(declare-fun Tclass._System.nat () T@U)
(declare-fun null () T@U)
(declare-fun Tclass._System.array (T@U) T@U)
(declare-fun Tclass._System.array_0 (T@U) T@U)
(declare-fun dtype (T@U) T@U)
(declare-fun Tclass._System.___hFunc0 (T@U) T@U)
(declare-fun Tclass._System.___hFunc0_0 (T@U) T@U)
(declare-fun HandleTypeType () T@T)
(declare-fun Apply0 (T@U T@U T@U) T@U)
(declare-fun Handle0 (T@U T@U T@U) T@U)
(declare-fun Requires0 (T@U T@U T@U) Bool)
(declare-fun Reads0 (T@U T@U T@U) T@U)
(declare-fun Tclass._System.___hPartialFunc0 (T@U) T@U)
(declare-fun Tclass._System.___hPartialFunc0_0 (T@U) T@U)
(declare-fun Tclass._System.___hTotalFunc0 (T@U) T@U)
(declare-fun Tclass._System.___hTotalFunc0_0 (T@U) T@U)
(declare-fun DatatypeCtorId (T@U) T@U)
(declare-fun |@sharp@_System._tuple@sharp@0._@sharp@Make0| () T@U)
(declare-fun _System.__tuple_h0.___hMake0_q (T@U) Bool)
(declare-fun Tclass._System.__tuple_h0 () T@U)
(declare-fun |$IsA@sharp@_System.__tuple_h0| (T@U) Bool)
(declare-fun |@sharp@_System._tuple@sharp@2._@sharp@Make2| (T@U T@U) T@U)
(declare-fun _System.__tuple_h2.___hMake2_q (T@U) Bool)
(declare-fun Tclass._System.__tuple_h2 (T@U T@U) T@U)
(declare-fun Tclass._System.__tuple_h2_0 (T@U) T@U)
(declare-fun Tclass._System.__tuple_h2_1 (T@U) T@U)
(declare-fun |$IsA@sharp@_System.__tuple_h2| (T@U) Bool)
(declare-fun Tclass._System.___hFunc1 (T@U T@U) T@U)
(declare-fun Tclass._System.___hFunc1_0 (T@U) T@U)
(declare-fun Tclass._System.___hFunc1_1 (T@U) T@U)
(declare-fun MapType2Type (T@T T@T T@T) T@T)
(declare-fun MapType2TypeInv0 (T@T) T@T)
(declare-fun MapType2TypeInv1 (T@T) T@T)
(declare-fun MapType2TypeInv2 (T@T) T@T)
(declare-fun MapType2Select (T@U T@U T@U) T@U)
(declare-fun MapType2Store (T@U T@U T@U T@U) T@U)
(declare-fun Apply1 (T@U T@U T@U T@U T@U) T@U)
(declare-fun Handle1 (T@U T@U T@U) T@U)
(declare-fun Requires1 (T@U T@U T@U T@U T@U) Bool)
(declare-fun Reads1 (T@U T@U T@U T@U T@U) T@U)
(declare-fun Tclass._System.___hPartialFunc1 (T@U T@U) T@U)
(declare-fun Tclass._System.___hPartialFunc1_0 (T@U) T@U)
(declare-fun Tclass._System.___hPartialFunc1_1 (T@U) T@U)
(declare-fun Tclass._System.___hTotalFunc1 (T@U T@U) T@U)
(declare-fun Tclass._System.___hTotalFunc1_0 (T@U) T@U)
(declare-fun Tclass._System.___hTotalFunc1_1 (T@U) T@U)
(declare-fun Tclass._module.__default () T@U)
(declare-fun MapType3Type (T@T T@T) T@T)
(declare-fun MapType3TypeInv0 (T@T) T@T)
(declare-fun MapType3TypeInv1 (T@T) T@T)
(declare-fun MapType3Select (T@U T@U T@U) T@U)
(declare-fun MapType3Store (T@U T@U T@U T@U) T@U)
(declare-fun |lambda@sharp@0| (T@U T@U T@U Bool) T@U)
(declare-fun |m@sharp@0@0| () T@U)
(declare-fun $_Frame@0 () T@U)
(declare-fun $Heap@@15 () T@U)
(declare-fun |m@sharp@0| () T@U)
(declare-fun %lbl%+0 () Bool)
(declare-fun %lbl%@1 () Bool)
(declare-fun %lbl%@2 () Bool)
(declare-fun %lbl%@3 () Bool)
(declare-fun %lbl%@4 () Bool)
(declare-fun %lbl%+5 () Bool)
(declare-fun %lbl%+6 () Bool)
(declare-fun %lbl%+7 () Bool)
(declare-fun %lbl%@8 () Bool)
(declare-fun %lbl%@9 () Bool)
(declare-fun %lbl%+10 () Bool)
(declare-fun %lbl%+11 () Bool)
(declare-fun %lbl%+12 () Bool)
(declare-fun %lbl%@13 () Bool)
(declare-fun %lbl%@14 () Bool)
(declare-fun %lbl%+15 () Bool)
(declare-fun $IsHeapAnchor (T@U) Bool)
(declare-fun $FunctionContextHeight () Int)
(declare-fun bx@@10!335!0 (T@U T@U) T@U)
(declare-fun bx@@12!337!1 (T@U T@U) T@U)
(declare-fun bx@@14!339!2 (T@U T@U) T@U)
(declare-fun i@@0!342!3 (T@U T@U) Int)
(declare-fun bx@@16!344!4 (T@U T@U T@U) T@U)
(declare-fun bx@@18!346!5 (T@U T@U T@U) T@U)
(declare-fun bx@@20!348!6 (T@U T@U T@U) T@U)
(declare-fun i@@2!350!7 (T@U T@U T@U) Int)
(declare-fun bx@@22!352!8 (T@U T@U T@U) T@U)
(declare-fun bx@@24!354!9 (T@U T@U T@U T@U) T@U)
(declare-fun bx@@26!356!10 (T@U T@U T@U) T@U)
(declare-fun bx@@28!358!11 (T@U T@U T@U T@U) T@U)
(declare-fun x@@16!389!12 (T@U) T@U)
(declare-fun o@@9!412!13 (T@U T@U) T@U)
(declare-fun o@@11!414!14 (T@U T@U) T@U)
(declare-fun o@@13!417!15 (T@U T@U) T@U)
(declare-fun o@@20!434!16 (T@U T@U) T@U)
(declare-fun o@@22!436!17 (T@U T@U) T@U)
(declare-fun o@@24!439!18 (T@U T@U) T@U)
(declare-fun bx@@32!446!19 (T@U) T@U)
(declare-fun x@@26!451!20 (T@U) T@U)
(declare-fun o@@32!468!21 (T@U T@U) T@U)
(declare-fun o@@34!470!22 (T@U T@U) T@U)
(declare-fun o@@36!473!23 (T@U T@U) T@U)
(declare-fun i@@10!483!24 (T@U T@U) Int)
(declare-fun i@@14!500!25 (T@U T@U) Int)
(declare-fun i@@16!505!26 (T@U Int T@U) Int)
(declare-fun i@@18!507!27 (T@U Int T@U) Int)
(declare-fun j@@0!509!28 (T@U T@U) Int)
(declare-fun j@@2!512!29 (Int T@U T@U) Int)
(declare-fun i@@21!523!30 (T@U T@U T@U) Int)
(declare-fun u@@5!542!31 (T@U T@U) T@U)
(declare-fun x@@40!547!32 (T@U) T@U)
(declare-fun u@@15!557!34 (T@U T@U) T@U)
(declare-fun u@@14!556!33 (T@U T@U) T@U)
(declare-fun o@@38!560!35 (T@U T@U) T@U)
(declare-fun u@@16!562!36 (T@U T@U) T@U)
(declare-fun u@@24!572!38 (T@U T@U) T@U)
(declare-fun u@@23!571!37 (T@U T@U) T@U)
(declare-fun fld!603!39 (T@U T@U T@U T@U) T@U)
(declare-fun o@@39!603!40 (T@U T@U T@U T@U) T@U)
(declare-fun o@@40!605!42 (T@U T@U T@U T@U) T@U)
(declare-fun fld@@0!605!41 (T@U T@U T@U T@U) T@U)
(declare-fun o@@41!607!44 (T@U T@U T@U T@U) T@U)
(declare-fun fld@@1!607!43 (T@U T@U T@U T@U) T@U)
(declare-fun o@@42!609!46 (T@U T@U T@U T@U) T@U)
(declare-fun fld@@2!609!45 (T@U T@U T@U T@U) T@U)
(declare-fun fld@@3!611!47 (T@U T@U T@U T@U) T@U)
(declare-fun o@@43!611!48 (T@U T@U T@U T@U) T@U)
(declare-fun fld@@4!613!49 (T@U T@U T@U T@U) T@U)
(declare-fun o@@44!613!50 (T@U T@U T@U T@U) T@U)
(declare-fun h@@23!615!51 (T@U T@U) T@U)
(declare-fun bx@@39!617!52 (T@U T@U) T@U)
(declare-fun r@@9!619!53 (T@U T@U T@U) T@U)
(declare-fun $Heap!625!54 (T@U T@U) T@U)
(declare-fun $Heap@@2!626!55 (T@U T@U) T@U)
(declare-fun $Heap@@3!632!56 (T@U T@U) T@U)
(declare-fun $Heap@@6!633!57 (T@U T@U) T@U)
(declare-fun a@sharp@6@sharp@0@sharp@0!644!59 (T@U) T@U)
(declare-fun a@sharp@6@sharp@1@sharp@0!644!58 (T@U) T@U)
(declare-fun fld@@5!666!60 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun o@@45!666!61 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun fld@@6!668!62 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun o@@46!668!63 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun fld@@7!670!64 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun o@@47!670!65 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun o@@48!672!67 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun fld@@8!672!66 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun fld@@9!674!68 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun o@@49!674!69 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun o@@50!676!71 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun fld@@10!676!70 (T@U T@U T@U T@U T@U T@U) T@U)
(declare-fun bx0@@9!678!72 (T@U T@U T@U) T@U)
(declare-fun h@@30!678!73 (T@U T@U T@U) T@U)
(declare-fun bx@@47!681!75 (T@U T@U) T@U)
(declare-fun bx@@46!680!74 (T@U T@U) T@U)
(declare-fun r@@14!683!77 (T@U T@U T@U T@U) T@U)
(declare-fun bx0@@11!684!76 (T@U T@U T@U T@U) T@U)
(declare-fun $Heap@@10!694!80 (T@U T@U T@U) T@U)
(declare-fun x0@sharp@0@@2!692!81 (T@U T@U T@U) T@U)
(declare-fun $Heap@@7!693!78 (T@U T@U T@U) T@U)
(declare-fun x0@sharp@0@@1!692!79 (T@U T@U T@U T@U) T@U)
(declare-fun $Heap@@11!702!82 (T@U T@U T@U) T@U)
(declare-fun $Heap@@14!703!84 (T@U T@U T@U) T@U)
(declare-fun x0@sharp@0@@5!701!83 (T@U T@U T@U T@U) T@U)
(declare-fun x0@sharp@0@@6!701!85 (T@U T@U T@U) T@U)
(assert (! (forall ((arg0py0 Int)) (! (= (U_2_int (int_2_U arg0py0)) arg0py0) :pattern ((int_2_U arg0py0)) )) :named A1_4))
(assert (! (forall ((arg0@@0py0 Int)) (! (= (type (int_2_U arg0@@0py0)) intType) :pattern ((int_2_U arg0@@0py0)) )) :named A1_5))
(assert (! (forall ((arg0@@36py0 T@T)(arg1@@7py0 T@T)) (! (= (MapTypeInv0 (MapType arg0@@36py0 arg1@@7py0)) arg0@@36py0) :pattern ((MapType arg0@@36py0 arg1@@7py0)) )) :named A57_1))
(assert (! (forall ((arg0@@37py0 T@T)(arg1@@8py0 T@T)) (! (= (MapTypeInv1 (MapType arg0@@37py0 arg1@@8py0)) arg1@@8py0) :pattern ((MapType arg0@@37py0 arg1@@8py0)) )) :named A57_2))
(assert (! (forall ((a@@2py0 T@U)(x@@18py0 T@U)) (! (or (U_2_bool (MapType0Select (Set@sharp@UnionOne a@@2py0 x@@18py0) x@@18py0)) (not (= (type a@@2py0) (MapType0Type (type x@@18py0) boolType)))) :pattern ((Set@sharp@UnionOne a@@2py0 x@@18py0)) )) :named A131))
(assert (! (forall ((a@@4py0 T@U)(x@@20py0 T@U)) (! (or (not (U_2_bool (MapType0Select a@@4py0 x@@20py0))) (= (Set@sharp@Card (Set@sharp@UnionOne a@@4py0 x@@20py0)) (Set@sharp@Card a@@4py0)) (not (= (type a@@4py0) (MapType0Type (type x@@20py0) boolType)))) :pattern ((Set@sharp@Card (Set@sharp@UnionOne a@@4py0 x@@20py0))) )) :named A133))
(assert (! (forall ((arg0@@87py0 T@U)) (! (= (type (Map@sharp@Values arg0@@87py0)) (MapType0Type (MapTypeInv1 (type arg0@@87py0)) boolType)) :pattern ((Map@sharp@Values arg0@@87py0)) )) :named A283))
(assert (! (forall ((m@@9py0 T@U)) (! (or (not (= (type m@@9py0) (MapType (MapTypeInv0 (type m@@9py0)) (MapTypeInv1 (type m@@9py0))))) (= (Set@sharp@Card (Map@sharp@Values m@@9py0)) (Map@sharp@Card m@@9py0))) :pattern ((Set@sharp@Card (Map@sharp@Values m@@9py0))) )) :named A284))
(assert (! (forall ((m@@10py0 T@U)(v@@36py0 T@U)) (! (or (not (= (type m@@10py0) (MapType (MapTypeInv0 (type m@@10py0)) (type v@@36py0)))) (and (or (not (U_2_bool (MapType0Select (Map@sharp@Values m@@10py0) v@@36py0))) (and (= (type (u@@5!542!31 v@@36py0 m@@10py0)) (MapTypeInv0 (type m@@10py0))) (U_2_bool (MapType0Select (Map@sharp@Domain m@@10py0) (u@@5!542!31 v@@36py0 m@@10py0))) (= v@@36py0 (MapType0Select (Map@sharp@Elements m@@10py0) (u@@5!542!31 v@@36py0 m@@10py0))))) (or (U_2_bool (MapType0Select (Map@sharp@Values m@@10py0) v@@36py0)) (forall ((u@@6py0 T@U)) (! (or (not (= (type u@@6py0) (MapTypeInv0 (type m@@10py0)))) (not (U_2_bool (MapType0Select (Map@sharp@Domain m@@10py0) u@@6py0))) (not (= v@@36py0 (MapType0Select (Map@sharp@Elements m@@10py0) u@@6py0)))) :pattern ((MapType0Select (Map@sharp@Domain m@@10py0) u@@6py0)) :pattern ((MapType0Select (Map@sharp@Elements m@@10py0) u@@6py0)) ))))) :pattern ((MapType0Select (Map@sharp@Values m@@10py0) v@@36py0)) )) :named A285))
(assert (! (forall ((U@@8py0 T@T)(V@@7py0 T@T)) (! (= (type (Map@sharp@Empty U@@8py0 V@@7py0)) (MapType U@@8py0 V@@7py0)) :pattern ((Map@sharp@Empty U@@8py0 V@@7py0)) )) :named A290))
(assert (! (forall ((u@@7py0 T@U)(V@@8py0 T@T)) (! (not (U_2_bool (MapType0Select (Map@sharp@Domain (Map@sharp@Empty (type u@@7py0) V@@8py0)) u@@7py0))) :pattern ((MapType0Select (Map@sharp@Domain (Map@sharp@Empty (type u@@7py0) V@@8py0)) u@@7py0)) )) :named A291))
(assert (! (forall ((arg0@@92py0 T@U)(arg1@@38py0 T@U)(arg2@@4py0 T@U)) (! (= (type (Map@sharp@Build arg0@@92py0 arg1@@38py0 arg2@@4py0)) (MapType (type arg1@@38py0) (type arg2@@4py0))) :pattern ((Map@sharp@Build arg0@@92py0 arg1@@38py0 arg2@@4py0)) )) :named A297))
(assert (! (forall ((m@@14py0 T@U)(u@@8py0 T@U)(u@quote@py0 T@U)(v@@37py0 T@U)) (! (or (and (or (not (= u@quote@py0 u@@8py0)) (and (U_2_bool (MapType0Select (Map@sharp@Domain (Map@sharp@Build m@@14py0 u@@8py0 v@@37py0)) u@quote@py0)) (= (MapType0Select (Map@sharp@Elements (Map@sharp@Build m@@14py0 u@@8py0 v@@37py0)) u@quote@py0) v@@37py0))) (or (= u@quote@py0 u@@8py0) (and (or (U_2_bool (MapType0Select (Map@sharp@Domain m@@14py0) u@quote@py0)) (not (U_2_bool (MapType0Select (Map@sharp@Domain (Map@sharp@Build m@@14py0 u@@8py0 v@@37py0)) u@quote@py0)))) (or (U_2_bool (MapType0Select (Map@sharp@Domain (Map@sharp@Build m@@14py0 u@@8py0 v@@37py0)) u@quote@py0)) (not (U_2_bool (MapType0Select (Map@sharp@Domain m@@14py0) u@quote@py0)))) (= (MapType0Select (Map@sharp@Elements (Map@sharp@Build m@@14py0 u@@8py0 v@@37py0)) u@quote@py0) (MapType0Select (Map@sharp@Elements m@@14py0) u@quote@py0))))) (not (= (type m@@14py0) (MapType (type u@@8py0) (type v@@37py0)))) (not (= (type u@quote@py0) (type u@@8py0)))) :pattern ((MapType0Select (Map@sharp@Domain (Map@sharp@Build m@@14py0 u@@8py0 v@@37py0)) u@quote@py0)) :pattern ((MapType0Select (Map@sharp@Elements (Map@sharp@Build m@@14py0 u@@8py0 v@@37py0)) u@quote@py0)) )) :named A298))
(assert (! (forall ((m@@16py0 T@U)(u@@10py0 T@U)(v@@39py0 T@U)) (! (or (U_2_bool (MapType0Select (Map@sharp@Domain m@@16py0) u@@10py0)) (not (= (type m@@16py0) (MapType (type u@@10py0) (type v@@39py0)))) (= (Map@sharp@Card (Map@sharp@Build m@@16py0 u@@10py0 v@@39py0)) (+ 1 (Map@sharp@Card m@@16py0)))) :pattern ((Map@sharp@Card (Map@sharp@Build m@@16py0 u@@10py0 v@@39py0))) )) :named A300))
(assert (! (forall ((m@@17py0 T@U)(u@@11py0 T@U)(v@@40py0 T@U)) (! (or (not (= (type m@@17py0) (MapType (type u@@11py0) (type v@@40py0)))) (= (Map@sharp@Values (Map@sharp@Build m@@17py0 u@@11py0 v@@40py0)) (Set@sharp@UnionOne (Map@sharp@Values m@@17py0) v@@40py0))) :pattern ((Map@sharp@Values (Map@sharp@Build m@@17py0 u@@11py0 v@@40py0))) )) :named A301))

;(declare-fun dummy (Bool) Bool)
;(assert (! (dummy (= (Set@sharp@Card (Map@sharp@Values (Map@sharp@Build (Map@sharp@Build (Map@sharp@Empty intType intType) (int_2_U 1) (int_2_U 1)) (int_2_U 2) (int_2_U 1))))
;                     (Set@sharp@Card (Map@sharp@Domain (Map@sharp@Build (Map@sharp@Build (Map@sharp@Empty intType intType) (int_2_U 1) (int_2_U 1)) (int_2_U 2) (int_2_U 1)))))) :named Repro2))

(check-sat)
(get-info :reason-unknown)
;z3 -T:600 smt-inputs/benchmarks/verifiers/dafny/pattern_augmenter/ematching/tmp/dafny-issue135_std_unique_aug-gt_unsat-full.smt2
;unknown
;((:reason-unknown "smt tactic failed to show goal to be sat/unsat (incomplete quantifiers)"))
