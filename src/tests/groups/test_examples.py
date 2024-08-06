###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2024 ETH Zurich.
###############################################################################

from typing import List

from src.algorithms.groups import GroupsAxiomTester
from src.frontend.optset import Optset
from src.tests.base_unit_test import BaseUnitTest


# Note: Some of the tests may fail, as Z3 may generate different constants than the expect ones (the triggering terms
# are not unique for many of the examples)

class TestExamples(BaseUnitTest):

    paper_dir = "../../smt-inputs/paper/"
    gobra_dir = "../../smt-inputs/benchmarks/verifiers/gobra/"
    gobra_small_dir = "../../smt-inputs/small/"

    def individual_file(self, file_name: str, optset: Optset=Optset(), expected_output: bool=True,
                        expected_dummies: List[str]=[], expected_sigma=0.3):
        algo = GroupsAxiomTester(file_name, opt=optset)
        inconsistent, dummies, final_sigma = algo.run()
        self.assertEqual(expected_output, inconsistent)
        if inconsistent:
            self.assertEqual(expected_sigma, final_sigma)
            self.assertListEqual(expected_dummies, dummies)

    def test_figure1(self):
        self.individual_file(self.paper_dir + "figure1.smt2",
                             optset=Optset(type_constraints=True),
                             expected_dummies=["(! (__dummy_A4T165@0__ (Seq@sharp@Length (Seq@sharp@Build (Seq@sharp@Build _0 7 _1 8) 9 _1 (- 1)))) :named Dummy_A4T165@0 )"])

    def test_figure2(self):
        self.individual_file(self.paper_dir + "figure2.smt2",
                             expected_dummies=["(! (__dummy_A0T0@0__ (f 0)) :named Dummy_A0T0@0 )"])

    def test_figure3(self):
        self.individual_file(self.paper_dir + "figure3.smt2", expected_output=False)

    def test_figure4(self):
        self.individual_file(self.paper_dir + "figure4.smt2",
                             expected_dummies=["(! (__dummy_A0T0@0__ ($div 7 (- 3))) :named Dummy_A0T0@0 )"])

    def test_figure4_fstar(self):
        self.individual_file(self.paper_dir + "figure4_fstar.smt2", optset=Optset(max_different_models=8),
                             expected_dummies=["(! (__dummy_A0T0@5__ (_div 1 2)) :named Dummy_A0T0@5 )"])

    def test_figure5(self):
        self.individual_file(self.paper_dir + "figure5.smt2",
                             expected_dummies=["(! (__dummy_A0T0@0__ (f 2)) :named Dummy_A0T0@0 )"])

    def test_figure6(self):
        self.individual_file(self.paper_dir + "figure6.smt2",
                             expected_dummies=["(! (__dummy_A0T1@0__ (f (g 7))) :named Dummy_A0T1@0 )"])

    def test_figure7(self):
        self.individual_file(self.paper_dir + "figure7.smt2",
                             expected_dummies=["(! (__dummy_A0T1@0__ (f (g x2))) :named Dummy_A0T1@0 )"])

    def test_figure8(self):
        self.individual_file(self.paper_dir + "figure8.smt2",
                             expected_dummies=["(! (__dummy_A0T1@0__ (f _0) (g _0)) :named Dummy_A0T1@0 )"])

    def test_figure9(self):
        self.individual_file(self.paper_dir + "figure9.smt2",
                             expected_dummies=["(! (__dummy_A2T39@2__ (sum empty (+ 0 1) 0) (sum empty 0 0)) :named Dummy_A2T39@2 )"],
                             expected_sigma=0.2)

    def test_figure10(self):
        self.individual_file(self.paper_dir + "figure10.smt2",
                             expected_dummies=["(! (__dummy_A0T2@0__ (f 7) (g _0)) :named Dummy_A0T2@0 )"])

    def test_figure11(self):
        self.individual_file(self.paper_dir + "figure11.smt2",
                             expected_dummies=["(! (__dummy_A0T1@0__ (contained EmptyList 0) (isEmpty EmptyList)) :named Dummy_A0T1@0 )"])

    def _test_figure12(self):
        self.individual_file(self.paper_dir + "figure12.smt2", expected_output=False)

    def test_figure13(self):
        self.individual_file(self.paper_dir + "figure13.smt2",
                             expected_dummies=["(! (__dummy_A0T0@0__ (P 2) (Q 3 4 2)) :named Dummy_A0T0@0 )"])

    def test_figure14(self):
        self.individual_file(self.paper_dir + "figure14.smt2",
                             expected_dummies=["(! (__dummy_A0T2@2__ (len (next 7))) :named Dummy_A0T2@2 )"])

    def test_figure15(self):
        self.individual_file(self.paper_dir + "figure15.smt2",
                             expected_dummies=["(! (__dummy_A0T1__ (Q k)) :named Dummy_A0T1 )"])

    def test_figure16(self):
        self.individual_file(self.paper_dir + "figure16.smt2",
                             expected_dummies=["(! (dummy_merged (g (+ 0 1)) (g 2)) :named Dummy_merged )"])

    def test_figure17(self):
        self.individual_file(self.paper_dir + "figure17.smt2", optset=Optset(top_level=False),
                             expected_dummies=["(! (dummy_merged (MapType1Select (MapType1Store _0 (MapType1Select (MapType1Store _0 QPMask@0 _0 QPMask@0) _1 _2) _0 QPMask@0) _1 _2) (MapType1Select (MapType1Store _0 (MapType1Select (MapType1Store _0 _2 _0 ZeroPMask) _1 _2) _0 ZeroPMask) _1 _2) (MapType1Select (MapType1Store _0 QPMask@0 _1 (MapType1Select (MapType1Store _0 QPMask@0 _1 _1) _3 _0)) _3 _0) (MapType1Select (MapType1Store _0 ZeroPMask _1 (MapType1Select (MapType1Store _0 ZeroPMask _1 _1) _3 _0)) _3 _0)) :named Dummy_merged )"],
                             expected_sigma=0.2)

    def test_figure18(self):
        self.individual_file(self.paper_dir + "figure18.smt2", optset=Optset(top_level=False),
                             expected_dummies=["(! (dummy_merged (MapType1Select (MapType1Store _2 (bool_2_U false) _1 (bool_2_U false)) _2 _3) (MapType1Select (MapType1Store _2 (int_2_U 2) _0 (int_2_U 2)) _3 _1) (MapType1Select (MapType1Store _0 (int_2_U 2) _1 (bool_2_U false)) _2 _1) (MapType1Select (MapType1Store _0 (int_2_U 2) _0 (int_2_U 2)) _1 _2) (MapType1Select (MapType1Store _3 (bool_2_U true) _2 (int_2_U 1)) _3 _1) (MapType1Select (MapType1Store _3 (bool_2_U true) _3 (bool_2_U true)) _1 _0)) :named Dummy_merged )"],
                             expected_sigma=0.2)

    def test_figure19(self):
        self.individual_file(self.paper_dir + "figure19.smt2", optset=Optset(top_level=False),
                             expected_dummies=["(! (__dummy_A0T1__ (f (g 2020))) :named Dummy_A0T1 )"])

    def test_figure20(self):
        self.individual_file(self.paper_dir + "figure20.smt2",
                             expected_dummies=["(! (dummy_merged (g _0) (i (h _0))) :named Dummy_merged )"])

    def test_figure21(self):
        self.individual_file(self.paper_dir + "figure21.smt2",
                             expected_dummies=["(! (__dummy_A1T0@0__ (f a b)) :named Dummy_A1T0@0 )"])

    def test_figure22(self):
        self.individual_file(self.paper_dir + "figure22.smt2",
                             expected_dummies=["(! (__dummy_A0T0@0__ (f false false)) :named Dummy_A0T0@0 )"])

    def test_figure24(self):
        self.individual_file(self.paper_dir + "figure24.smt2",
                             expected_dummies=["(! (__dummy_A0T0@0__ (f 0 3)) :named Dummy_A0T0@0 )"])

    def test_figure25(self):
        self.individual_file(self.paper_dir + "figure25.smt2", expected_output=False)

    def test_arrayWithNil1(self):
        self.individual_file(self.gobra_dir + "arrayWithNil1_std_unique_aug-gt_unsat-full_incomplete-quant.smt2",
                             expected_dummies=["(! (dummy_merged (loc nil 0) (loc nil 1)) :named Dummy_merged )"])

    def test_arrayWithNil2(self):
        self.individual_file(self.gobra_dir + "arrayWithNil2_std_unique_aug-gt_unsat-full_incomplete-quant.smt2",
                             expected_dummies=["(! (dummy_merged (loc _0 (unbox17 _0 (nil (Emb17DomainTypeType _0))) 2) (loc _0 (unbox42 _0 (nil (Emb42DomainTypeType _0))) 1)) :named Dummy_merged )"])

    def test_arrayWithNil3(self):
        self.individual_file(self.gobra_dir + "arrayWithNil3_std_unique_aug-gt_unsat-full_incomplete-quant.smt2",
                             expected_dummies=["(! (dummy_merged (loc (unbox null) 0) (loc (unbox null) 1)) :named Dummy_merged )"])

    def test_arrayWithNil5(self):
        self.individual_file(self.gobra_dir + "arrayWithNil5_std_unique_aug-gt_unsat-full_incomplete-quant.smt2",
                             expected_dummies=["(! (dummy_merged (loc (unbox null) 0) (loc (unbox null) 2)) :named Dummy_merged )"])

    def test_arrayWithNil6(self):
        self.individual_file(self.gobra_dir + "arrayWithNil6_std_unique_aug-gt_unsat-full_incomplete-quant.smt2",
                             expected_dummies=["(! (dummy_merged (loc nil 0) (loc nil 2)) :named Dummy_merged )"])

    def test_arrayWithNil7(self):
        self.individual_file(self.gobra_dir + "arrayWithNil7_std_unique_aug-gt_unsat-full_incomplete-quant.smt2",
                             expected_dummies=["(! (dummy_merged (loc (unboxA nilA) 2) (loc (unboxB nilB) 0)) :named Dummy_merged )"])

    def test_arrayWithNil8(self):
        self.individual_file(self.gobra_dir + "arrayWithNil8_std_unique_aug-gt_unsat-full_incomplete-quant.smt2",
                             optset=Optset(similarity_threshold=0.1),
                             expected_dummies=["(! (dummy_merged (loc (unboxA (unboxEmbA null)) 2) (loc (unboxB (unboxEmbB null)) 0)) :named Dummy_merged )"],
                             expected_sigma=0.1)

    def test_arrayWithNil9(self):
        self.individual_file(self.gobra_dir + "arrayWithNil9_std_unique_aug-gt_unsat-full_incomplete-quant.smt2",
                             optset=Optset(similarity_threshold=0.1),
                             expected_dummies=["(! (dummy_merged (loc (unbox17 (unbox Emb17DomainTypeType null)) 1) (loc (unbox42 (unbox Emb42DomainTypeType null)) 2)) :named Dummy_merged )"],
                             expected_sigma=0.1)

    def test_arrayWithNil9_min(self):
        self.individual_file(self.gobra_small_dir + "arrayWithNil9_std_unique_aug-gt_unsat-full_incomplete-quant_min.smt2",
                             optset=Optset(similarity_threshold=0.1),
                             expected_dummies=["(! (dummy_merged (loc (unbox17 (unbox Emb17DomainTypeType null)) 2) (loc (unbox42 (unbox Emb42DomainTypeType null)) 2)) :named Dummy_merged )"],
                             expected_sigma=0.1)

    def test_arrayWithNil10(self):
        self.individual_file(self.gobra_dir + "arrayWithNil10_std_unique_aug-gt_unsat-full_incomplete-quant.smt2",
                             optset=Optset(similarity_threshold=0.1),
                             expected_dummies=["(! (dummy_merged (loc (unbox17 (unbox Emb17DomainTypeType null)) 1) (loc (unbox42 (unbox Emb42DomainTypeType null)) 2) (loc (unbox77 (unbox Emb77DomainTypeType null)) 2)) :named Dummy_merged )"],
                             expected_sigma=0.1)

    def test_arrayWithNil10_min(self):
        self.individual_file(self.gobra_small_dir + "arrayWithNil10_std_unique_aug-gt_unsat-full_incomplete-quant_min.smt2",
                             optset=Optset(similarity_threshold=0.1),
                             expected_dummies=["(! (dummy_merged (loc (unbox17 (unbox Emb17DomainTypeType null)) 2) (loc (unbox42 (unbox Emb42DomainTypeType null)) 2) (loc (unbox77 (unbox Emb77DomainTypeType null)) 2)) :named Dummy_merged )"],
                             expected_sigma=0.1)

    def test_arrayWithNil11(self):
        self.individual_file(self.gobra_dir + "arrayWithNil11_std_unique_aug-gt_unsat-full_incomplete-quant.smt2",
                             optset=Optset(similarity_threshold=0.1),
                             expected_dummies=["(! (dummy_merged (loc (unbox17 (unbox Emb17DomainTypeType null)) 2) (loc (unbox42 (unbox Emb42DomainTypeType null)) 2) (loc (unbox77 (unbox Emb77DomainTypeType null)) 2)) :named Dummy_merged )"],
                             expected_sigma=0.1)

    def test_option_wrong(self):
        self.individual_file(self.gobra_dir + "option-wrong_std_unique_aug-gt_unsat-full_incomplete-quant.smt2",
                             optset=Optset(type_constraints=True),
                             expected_dummies=["(! (__dummy_A30_0T45@0__ (optsome _0 (optget _0 (optnone _0)))) :named Dummy_A30_0T45@0 )"])

    def test_option_wrong_min(self):
        self.individual_file(self.gobra_small_dir + "option-wrong_std_unique_aug-gt_unsat-full_incomplete-quant_min.smt2",
                             optset=Optset(type_constraints=True),
                             expected_dummies=["(! (__dummy_A30_0T43@0__ (optsome _0 (optget _0 (optnone _0)))) :named Dummy_A30_0T43@0 )"])


