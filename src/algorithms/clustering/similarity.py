###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

from collections import OrderedDict
from typing import Set, List, Dict, Tuple

from datasketch import MinHash, MinHashLSH
from sortedcontainers import SortedSet

from pysmt.fnode import FNode
from pysmt.typing import PySMTType
from src.algorithms.clustering.cluster import Cluster
from src.algorithms.formulas.axiom import Axiom


class SimilarityIndex:

    def __init__(self, axioms: List[Axiom], threshold: float, seed: int = 0, permutations: int = 128):
        self.threshold = threshold
        self.axioms = axioms

        self.ids_to_indexes: Dict[str, int] = OrderedDict()
        self.ids_to_axioms: Dict[str, Axiom] = OrderedDict()

        self.data = self.generate_data()

        self.min_hashes = []

        for row in self.data:
            current_hashes = MinHash(num_perm=permutations, seed=seed, hashfunc=SimilarityIndex.hash_function)
            current_hashes.update(0)  # unique positive number added to all rows such that they contain at least
                                      # one value
            for index in row:
                current_hashes.update(index + 1)  # we encode the relevant attributes from 1
                                                  # (0 is used as the default value, see above)

            self.min_hashes.append(current_hashes)

        self.lsh = MinHashLSH(threshold=threshold, num_perm=permutations)
        for index, _ in enumerate(self.data):
            name = self.axioms[index].axiom_id
            min_hash = self.min_hashes[index]
            self.lsh.insert(name, min_hash)

    @staticmethod
    def hash_function(data_point: int):
        return data_point

    @staticmethod
    def index_of(element, all_elements) -> int:
        try:
            index = all_elements.index(element)
            return index
        except ValueError:
            return -1

    @staticmethod
    def get_name(attribute):
        if isinstance(attribute, PySMTType):
            return attribute.name
        if isinstance(attribute, FNode):
            return attribute.symbol_name()
        raise Exception

    def add_attribute_indexes(self, data, line, axiom_attributes, all_attributes, offset=0):
        for attribute in axiom_attributes:
            name = self.get_name(attribute)
            index = self.index_of(name, all_attributes)
            if index == -1:
                data[line].add(offset + len(all_attributes))
                all_attributes.append(name)
            else:
                data[line].add(offset + index)

    def generate_data(self):
        lines = len(self.axioms)
        function_symbols: List[str] = []

        data = [SortedSet(key=lambda x: x) for _ in range(lines)]
        for line, axiom in enumerate(self.axioms):
            self.ids_to_indexes[axiom.axiom_id] = line
            self.ids_to_axioms[axiom.axiom_id] = axiom
            self.add_attribute_indexes(data, line, axiom.function_symbols.union(axiom.function_symbols_triggers),
                                       function_symbols)
        return data

    def get_similar_axioms_ids(self, axiom: Axiom) -> Set[str]:
        axiom_id = axiom.axiom_id
        axiom_number = self.ids_to_indexes[axiom_id]
        approximate_result = self.lsh.query(self.min_hashes[axiom_number])

        exact_result: Set[str] = SortedSet(key=lambda x: x)
        for other_axiom_id in approximate_result:
            if other_axiom_id == axiom_id:
                exact_result.add(other_axiom_id)
            else:
                other_axiom_number = self.ids_to_indexes[other_axiom_id]
                features_set_axiom = self.data[axiom_number]
                features_set_other_axiom = self.data[other_axiom_number]
                exact_similarity_index = self.compute_exact_similarity_index(features_set_axiom,
                                                                             features_set_other_axiom)
                if exact_similarity_index >= self.threshold:
                    exact_result.add(other_axiom_id)
        return exact_result

    def get_similar_axioms(self, axiom: Axiom) -> Set[Axiom]:
        similar_axioms_ids = self.get_similar_axioms_ids(axiom)
        similar_axioms: Set[Axiom] = SortedSet([self.ids_to_axioms[axiom_id] for axiom_id in similar_axioms_ids],
                                               key=lambda x: x.axiom_id)
        similar_axioms.remove(axiom)
        return similar_axioms

    def compute_exact_similarity_index(self, set1: SortedSet, set2: SortedSet) -> float:
        if len(set1) == 0 and len(set2) == 0:
            return 0
        num_common_elements, num_all_elements = self.num_intersect_and_union(set1, set2)
        return num_common_elements / num_all_elements

    def get_clusters_of_similar_axioms(self, transitive=True) -> Set[Cluster]:
        axioms_clusters: Set[Cluster] = SortedSet(key=lambda x: x.size)
        for axiom in self.axioms:
            similar_axioms: Set[Axiom] = self.get_similar_axioms(axiom)
            axiom.new_similar_axioms = not (axiom.axioms_with_common_symbols == similar_axioms)
            axiom.axioms_with_common_symbols = similar_axioms
            cluster_axioms: Set[Axiom] = SortedSet(similar_axioms.union({axiom}), key=lambda x: x.axiom_id)
            cluster: Cluster = Cluster(cluster_axioms)
            if len(axioms_clusters) == 0 or len(similar_axioms) == 0 or not transitive:
                axioms_clusters.add(cluster)
            else:
                # merge the already existing clusters that contain at least one of the similar axioms
                self.merge_clusters(axioms_clusters, cluster)

        for cluster_id, cluster in enumerate(axioms_clusters):
            for axiom in cluster.similar_axioms:
                axiom.cluster_id = cluster_id
        return axioms_clusters

    @staticmethod
    def merge_clusters(clusters: Set[Cluster], new_cluster: Cluster):
        axioms_ids = new_cluster.ids
        to_be_merged = [cluster for cluster in clusters if len(cluster.ids.intersection(axioms_ids)) > 0]

        if len(to_be_merged) > 0:
            merged: Set[Axiom] = SortedSet(key=lambda x: x.axiom_id)
            for cluster in to_be_merged:
                clusters.remove(cluster)
                merged.update(cluster.similar_axioms)
            clusters.add(Cluster(merged.union(new_cluster.similar_axioms)))
        else:
            clusters.add(new_cluster)

    # Efficiently counts the number of elements in the intersection and the union of two sorted input sets.
    # The result is equal to Tuple[len(set1.union(set2)), len(set1.intersect(set2))]
    @staticmethod
    def num_intersect_and_union(set1: SortedSet, set2: SortedSet) -> Tuple[int, int]:
        len1 = len(set1)
        len2 = len(set2)

        if len1 == 0:
            return 0, len2
        if len2 == 0:
            return 0, len1

        i = 0
        j = 0

        iter1 = iter(set1)
        iter2 = iter(set2)

        element1 = next(iter1)
        element2 = next(iter2)

        num_common_elements = 0
        num_all_elements = 0

        while i < len1 and j < len2:
            num_all_elements += 1
            if element1 < element2:
                i += 1
                if i < len1:
                    element1 = next(iter1)
            elif element2 < element1:
                j += 1
                if j < len2:
                    element2 = next(iter2)
            else:
                i += 1
                j += 1
                if i < len1:
                    element1 = next(iter1)
                if j < len2:
                    element2 = next(iter2)
                num_common_elements += 1

        # add remaining elements if one set is larger
        num_all_elements += len1 - i
        num_all_elements += len2 - j
        return num_common_elements, num_all_elements
