import os
import numpy as np

def test_finding_subgraph(recompute=True, del_output=True):
    if recompute:
        os.system("python finding_subgraphs.py Datasets/molecules-small.pos Datasets/molecules-small.pos 5 5 > fs_test.tmp")
    
    with open('fs_test.tmp', 'r') as test:
        test_lines = [line.strip() for line in test.readlines()]
        with open('Examples/task1_small_5_5.txt', 'r') as real:
            real_lines = [line.strip() for line in real.readlines()]

    print(f"\033[1;31;1mSolution has \t\t{len(real_lines)} lines,\n   while we have \t{len(test_lines)} lines\n\033[0;0;0m")

    wrong_lines = [line not in real_lines for line in test_lines]
    missing_lines = [line not in test_lines for line in real_lines]

    wrong_lines_i = [(i if wrong_lines[i] else None) for i in range(len(wrong_lines))]
    missing_lines_i = [(i if missing_lines[i] else None) for i in range(len(missing_lines))]

    
    print(f"\033[1;31;1mWrong lines   : {sum(wrong_lines)}\033[0;0;0m\t(present in output but not in solution)")
    print(f"\t{[i for i in wrong_lines_i if i is not None]}\n")
    print(f"\033[1;31;1mMissing lines : {sum(missing_lines)}\033[0;0;0m\t(present in solution but missing in output)")
    print(f"\t{[i for i in missing_lines_i if i is not None]}\n")

    if del_output:
        os.system("del fs_test.tmp")

if __name__ == '__main__':
    test_finding_subgraph(recompute=True, del_output=True)
