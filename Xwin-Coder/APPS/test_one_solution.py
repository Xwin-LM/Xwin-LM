"""
Run solutions from one problem.
"""
import argparse
import json
import numpy as np
import os
import pprint
import multiprocessing
import testing_util as test_util

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from types import SimpleNamespace
from typing import Dict


EXAMPLE_RESULTS = {"0": [[-2]],"1": [[False,False,False]],"2": [[True,True]],"3": [[False,True,False,True,False,False,False,True,False,True,False,True,True,True,False,True]],"4": [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]}
EXAMPLE_ARGS = SimpleNamespace(debug=True)
TIMEOUT = 10

def A(k: int):
    assert k>=0
    if k==0:
        return 1
    r = 1
    for i in range(1, k+1):
        r*=i
    return r

def C(m,k):
    assert m>=0 and k>0
    if k>m:
        return 0
    return A(m)/(A(k)*A(m-k))


def get_pass_k_for_this_problem(num_correct, num_solution, passk):
    """
    calculate pass k like humaneval for robust estimation
    """
    res = dict()
    for k in passk:
        res[k] = 1 - C(num_solution-num_correct, k)/C(num_solution, k)
    return res

def print_results(results: Dict, args:argparse.Namespace=None):
    """
    Given the results evaluated against the testcases we output some statistics.
    """
    res = []
    per_prob_res = []

    solution_num = len(list(results.values())[0])
    pass_k = [1,5,10,50,100,1000]
    pass_k = [i for i in pass_k if i <= solution_num]

    for index in results:
        res.extend(results[index])
        for solution in results[index]:
            problem_results = np.asarray(solution)
            per_prob_res.append(np.mean(problem_results > 0))
            # all_correct.append(np.all(problem_results > 0))

    # test pass_k, we test top k like humaneval
    pass_k_corrects = {i: [] for i in pass_k}
    for index in results:
        all_correct_for_this_problem = []
        assert solution_num == len(results[index])
        for solution in results[index]:
            problem_results = np.asarray(solution)
            all_correct_for_this_problem.append(np.all(problem_results > 0))
        num_correct_for_this_problem = sum(all_correct_for_this_problem)
        pkft = get_pass_k_for_this_problem(num_correct_for_this_problem, solution_num, pass_k)
        for i in pass_k:
            pass_k_corrects[i].append(pkft[i])


    # We count both compile errors and runtime errors for multiple tests as one error.

    compile_errors = len([e for e in res if -2 in e])
    runtime_errors = len([e for e in res if -1 in e])
    total_testcases = len(res)
    print(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases }")
    print(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}")
    print(f"number of test cases run = {total_testcases}")

    print(f"Test Case Average (average accuracy over problems) = {np.mean(per_prob_res)}")
    for k in pass_k:
        print(f"Strict Accuracy (all test cases passed / total problems) pass@{k} acc = {np.mean(pass_k_corrects[k])}")
    # print(f"Strict Accuracy (all test cases passed / total problems) = {np.mean(all_correct)}")


def check_correctness(prob_path, generation, timeout, debug):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(prob_path, generation, debug, result):
        result.append(test_util.run_test(prob_path=prob_path, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(prob_path, generation, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        # Reamark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead 
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
        if debug:
            print(f"global timeout")
    return result[0]


def eval_and_save_problems(args):
    with open(args.test_loc, "r") as f:
        problems = sorted(json.load(f))

    print(len(problems))
    gpt_codes = {}
    gpt_bleu = {}
    gpt_codebleu = {}
    results = {}
    codes_loc = os.path.join(args.save, f"all_codes.json")
    if not os.path.exists(codes_loc):
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")
        results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json") 
    else:
        results_loc = os.path.join(args.save, f"all_results.json") 

    with open(codes_loc, "r") as f: 
        gpt_codes = json.load(f)

    if args.index:
        problems = [problems[args.index]]
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = problems[start:end]

    if args.stop_early:
        problems = problems[:args.stop_early]

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        try:
            if args.debug:
                print(f"\n\nproblem path = {problem}")
            output_str = gpt_codes[str(index+args.start)]
        except:
            print("CANNOT FIND OUTPUT_STR FOR", problem)
            continue
        prob_path = os.path.join(args.root, problem)
        
        if not os.path.exists(args.save):
            os.makedirs(args.save)

        res = []
        for o_idx, o in enumerate(output_str):
            if args.debug:
                print(f"\nTesting solution {o_idx}")
            curr_res = [-2]
            try:
                curr_res = check_correctness(prob_path=prob_path, generation=o, timeout=TIMEOUT, debug=args.debug)
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                       e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    print(f"Results were not all True: {curr_res}")
            except Exception as e:
                print(f"test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)

        if args.debug:
            print(f"\nHow to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
            #print(f"results = {res}")
 
        results[index+args.start+args.index] = res
        
        with open(results_loc, "w") as f:
            try:
                f.write(json.dumps(results))
            except Exception as e:
                import pdb; pdb.set_trace()
                print("didn't save problem due to {e}")

    return results


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    with open(args.test_loc, "r") as f:
        n_problems = len(json.load(f))
    # change ratio to number
    args.start = int(args.start_ratio*n_problems)
    args.end = int(args.end_ratio*n_problems)

    if args.print_results:
        results = {}
        codes_loc = os.path.join(args.save, f"all_codes.json")
        if os.path.exists(codes_loc):
            results_loc = os.path.join(args.save, f"all_results.json") 
        else:
            results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json") 
        with open(results_loc, "r") as f: 
            results = json.load(f)
    else:
        results = eval_and_save_problems(args)

    print_results(results, args)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("-t","--test_loc", default="../data_split/test.json", type=str, help="path to the json containing problem paths to be evaluated.")
    parser.add_argument("-r","--root", default="./", type=str, help="where the data is stored.")
    parser.add_argument("-s","--start_ratio", default=0.0, type=float)
    parser.add_argument("-e","--end_ratio", default=1.0, type=float, help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.")
    parser.add_argument("-i", "--index", default=0, type=int)
    parser.add_argument("-p", "--print_results", action="store_true", help="If you have already evaluated the results and only want to print them.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results", help="Where the evaluated data is loaded from and results saved to.")
    parser.add_argument("--stop-early", default=None, type=int)
 
    args = parser.parse_args()

    main(args)
