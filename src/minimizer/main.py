###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import argparse
import os
from enum import Enum
from os import listdir
from os.path import isfile, join, basename, dirname

from timeout_decorator import timeout_decorator

from src.minimizer.e_matching_runner import EmatchingRunner
from src.algorithms.pattern_augmenter import PatternAugmenter
from src.minimizer.post_processor import PostProcessor
from src.minimizer.exceptions import InvalidSmtFileException, MinimizerException
from src.minimizer.proof_finder import ProofFinder
from src.minimizer.unsat_core_finder import UnsatCoreFinder
from src.minimizer.vampire import VampireRunner
from src.utils.enums import SMTSolver
from src.utils.file import ensure_dir_structure

DEBUG_MODE = True


class Pipeline(Enum):
    EMATCHING = 'ematching'
    PATTERN_AUGMENTER = 'pattern_augmenter'
    POST_PROCESSOR = 'post_processor'
    VAMPIRE_RUNNER = 'run_vampire'
    UNSAT_CORE_MBQI = 'unsat_core_mbqi'
    UNSAT_CORE_EMATCHING = 'unsat_core_ematching'
    PROOF_MBQI = 'proof_mbqi'
    PROOF_EMATCHING = 'proof_ematching'

    def __str__(self):
        return self.value


def main():
    global DEBUG_MODE

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('pipeline', type=str, choices=[str(choice) for choice in Pipeline],
                            help="Which pipeline to run")
    arg_parser.add_argument('--solver', type=str, choices=[str(choice) for choice in SMTSolver], default="z3",
                            help="Which solver to use")
    arg_parser.add_argument('--timeout', type=int, default=300, help='Hard timeout per file')
    arg_parser.add_argument('--location', type=str, default='.', help='Directory with SMT files')
    arg_parser.add_argument('--is_debug', type=lambda x: (str(x).lower() == 'true'), default=DEBUG_MODE,
                            help="Whether to run the pipeline in debug mode")
    arg_parser.add_argument('--is_carantine', type=lambda x: (str(x).lower() == 'true'), default=False,
                            help="Whether the pipeline is run on previously carantined files")
    arg_parser.add_argument('--enumerative', type=bool, default=False,
                            help='Use enumerative instantiation when E-matching saturates (for CVC4)')
    args = arg_parser.parse_args()

    DEBUG_MODE = args.is_debug

    location = args.location
    if args.is_carantine:
        carantine_dir = location
    else:
        carantine_dir = None

    if args.solver == str(SMTSolver.Z3):
        solver = SMTSolver(SMTSolver.Z3)
    elif args.solver == str(SMTSolver.CVC4):
        solver = SMTSolver(SMTSolver.CVC4)
    elif args.solver == str(SMTSolver.VAMPIRE):
        solver = SMTSolver(SMTSolver.VAMPIRE)
    else:
        raise MinimizerException('selected invalid SMTSolver')

    enum_inst = args.enumerative

    if args.pipeline == str(Pipeline.EMATCHING):
        target_dir = join(location, "ematching") if not isfile(location) else \
            join(dirname(location), "ematching")
        run_all(input_path=location, target_dir=target_dir, solver=solver, ematching=True,
                carantine_dir=carantine_dir, timeout_per_file=args.timeout)

    elif args.pipeline == str(Pipeline.PATTERN_AUGMENTER):
        target_dir = join(location, args.pipeline) if not isfile(location) else \
            join(dirname(location), args.pipeline)
        run_all(input_path=location, target_dir=target_dir, solver=solver, pattern_augmenter=True,
                carantine_dir=carantine_dir, timeout_per_file=args.timeout)

    elif args.pipeline == str(Pipeline.POST_PROCESSOR):
        target_dir = join(location, "post_processor") if not isfile(location) else \
            join(dirname(location), "post_processor")
        run_all(input_path=location, target_dir=target_dir, solver=solver, post_processor=True,
                carantine_dir=carantine_dir, timeout_per_file=args.timeout)

    elif args.pipeline == str(Pipeline.VAMPIRE_RUNNER):
        target_dir = join(location, "vampire") if not isfile(location) else \
            join(dirname(location), "vampire")
        run_all(input_path=location, target_dir=target_dir, run_vampire=True,
                carantine_dir=carantine_dir, timeout_per_file=args.timeout)

    elif args.pipeline == str(Pipeline.UNSAT_CORE_MBQI):
        target_dir = join(location, "unsat_core_mbqi") if not isfile(location) else \
            join(dirname(location), "unsat_core_mbqi")
        run_all(input_path=location, target_dir=target_dir, solver=solver, unsat_core=True, with_mbqi=True,
                carantine_dir=carantine_dir, timeout_per_file=args.timeout)

    elif args.pipeline == str(Pipeline.UNSAT_CORE_EMATCHING):
        target_dir = join(location, "unsat_core_ematching") if not isfile(location) else \
            join(dirname(location), "unsat_core_ematching")
        run_all(input_path=location, target_dir=target_dir, solver=solver, enum_inst=enum_inst,
                unsat_core=True, with_mbqi=False, carantine_dir=carantine_dir, timeout_per_file=args.timeout)

    elif args.pipeline == str(Pipeline.PROOF_MBQI):
        target_dir = join(location, "proof_mbqi") if not isfile(location) else \
            join(dirname(location), "proof_mbqi")
        run_all(input_path=location, target_dir=target_dir, solver=solver, proof=True, with_mbqi=True,
                carantine_dir=carantine_dir, timeout_per_file=args.timeout)

    elif args.pipeline == str(Pipeline.PROOF_EMATCHING):
        target_dir = join(location, "proof_ematching") if not isfile(location) else \
            join(dirname(location), "proof_ematching")
        run_all(input_path=location, target_dir=target_dir, solver=solver, enum_inst=enum_inst,
                proof=True, with_mbqi=False, carantine_dir=carantine_dir, timeout_per_file=args.timeout)

    else:
        raise MinimizerException('selected invalid pipeline')


def run_all(input_path: str,
            target_dir,
            timeout_per_file,
            solver=SMTSolver(SMTSolver.Z3),
            carantine_dir=None,
            run_vampire=False,
            ematching=False,
            pattern_augmenter=False,
            post_processor=False,
            unsat_core=False,
            proof=False,
            enum_inst=False,
            with_mbqi=False):

    if isfile(input_path):
        input_file = input_path

        if ematching:
            EmatchingRunner(input_file, solver=solver, target_dir=target_dir,
                            timeout_per_file=timeout_per_file, debug_mode=DEBUG_MODE)

        elif pattern_augmenter:
            PatternAugmenter(filename=input_file, target_dir=target_dir, tmp_file_dir=target_dir, debug=DEBUG_MODE)

        elif post_processor:
            PostProcessor(input_file, solver=solver, target_dir=target_dir, timeout_per_file=timeout_per_file,
                          debug_mode=DEBUG_MODE)

        elif unsat_core:
            UnsatCoreFinder(input_file, target_dir=target_dir, solver=solver, enum_inst=enum_inst, mbqi=with_mbqi,
                            timeout_per_file=timeout_per_file, debug_mode=DEBUG_MODE)

        elif proof:
            ProofFinder(input_file, target_dir=target_dir, solver=solver, enum_inst=enum_inst, mbqi=with_mbqi,
                        timeout_per_file=timeout_per_file, debug_mode=DEBUG_MODE)

        elif run_vampire:
            VampireRunner(input_file, target_dir=target_dir)

    else:
        @timeout_decorator.timeout(timeout_per_file, timeout_exception=StopIteration)
        def TimedEmatchingRunner(*args, **kwargs):
            return EmatchingRunner(*args, **kwargs)

        @timeout_decorator.timeout(timeout_per_file, timeout_exception=StopIteration)
        def TimedPatternAugmenter(*args, **kwargs):
            return PatternAugmenter(*args, **kwargs)

        @timeout_decorator.timeout(timeout_per_file, timeout_exception=StopIteration)
        def TimedPostProcessor(*args, **kwargs):
            return PostProcessor(*args, **kwargs)

        @timeout_decorator.timeout(timeout_per_file, timeout_exception=StopIteration)
        def TimedVampireRunner(*args, **kwargs):
            return VampireRunner(*args, **kwargs)

        @timeout_decorator.timeout(timeout_per_file, timeout_exception=StopIteration)
        def TimedUnsatCoreFinder(*args, **kwargs):
            return UnsatCoreFinder(*args, **kwargs)

        @timeout_decorator.timeout(timeout_per_file, timeout_exception=StopIteration)
        def TimedProofFinder(*args, **kwargs):
            return ProofFinder(*args, **kwargs)

        if carantine_dir is None:
            carantine_dir = join(input_path, "carantine")
            returned_from_carantine = None
        else:
            returned_from_carantine = join(input_path, "..", "refurbished")

        all_files = [join(input_path, f) for f in listdir(input_path) if isfile(join(input_path, f))
                     and f.endswith("smt2")]
        num_files = len(all_files)

        processed_something = False
        for input_file_num, input_file in enumerate(sorted(all_files, key=os.path.getsize)):
            processed_something = True
            try:
                print("Processing SMT2 file %d / %d: %s" % (input_file_num+1, num_files, input_file))

                if ematching:
                    TimedEmatchingRunner(input_file, solver=solver, target_dir=target_dir,
                                         timeout_per_file=timeout_per_file, debug_mode=DEBUG_MODE)

                elif pattern_augmenter:
                    TimedPatternAugmenter(filename=input_file, target_dir=target_dir,
                                          tmp_file_dir=target_dir, debug=DEBUG_MODE)

                elif post_processor:
                    TimedPostProcessor(input_file, solver=solver, target_dir=target_dir,
                                       timeout_per_file=timeout_per_file, debug_mode=DEBUG_MODE)

                elif unsat_core:
                    TimedUnsatCoreFinder(input_file, target_dir=target_dir, solver=solver, enum_inst=enum_inst,
                                         mbqi=with_mbqi, timeout_per_file=timeout_per_file)

                elif proof:
                    TimedProofFinder(input_file, target_dir=target_dir, solver=solver, enum_inst=enum_inst,
                                     mbqi=with_mbqi, timeout_per_file=timeout_per_file)

                elif run_vampire:
                    TimedVampireRunner(input_file, target_dir=target_dir, debug_mode=DEBUG_MODE)

            except MinimizerException as e:
                print("   !!!    While processing " + input_file + " Minimizer raised exception: " + str(e))
            except UnicodeDecodeError as e:
                print("   !!!    Minimizer cannot decode file: " + input_file + ": " + str(e))
            except InvalidSmtFileException as e:
                print("   !!!    Skipping invalid SMT file: " + str(e))
            except NotImplementedError as e:
                print("   !!!    Not implemented, probably in PySMT: " + str(e))
            except StopIteration as _:
                print("   !!!    Minimizer timed out after %d seconds" % timeout_per_file)
            except Exception as e:
                if DEBUG_MODE:
                    raise e
                else:
                    print("   !!!    Something went terribly wrong while processing file `" +
                          input_file + ": " + str(e))
                    file_name = basename(input_file)
                    ensure_dir_structure(carantine_dir)
                    os.rename(input_file, join(carantine_dir, file_name))

            if input_path == carantine_dir:
                os.rename(input_file, join(returned_from_carantine, basename(input_file)))

        if not processed_something:
            print("Warning: no .smt2 files found in the location directory: `%s`" % input_path)

        print('Finished processing all files in ' + input_path)


if __name__ == '__main__':
    main()
