import os

def test_finding_subgraph(recompute=True, del_output=True):
    if recompute:
        return_value = os.system("python finding_subgraphs.py Datasets/molecules-small.pos Datasets/molecules-small.neg 5 5 > fs_test.tmp")
        if return_value != 0:
            exit(return_value)
    
    with open('fs_test.tmp', 'r') as test:
        test_lines = [line.strip() for line in test.readlines()]
    with open('Examples/task1_small_5_5.txt', 'r') as real:
        real_lines = [line.strip() for line in real.readlines()]

    red = "[1;31;1m"
    green = "[1;32;1m"

    color = green if len(real_lines) == len(test_lines) else red
    print(f"\033{color}Solution has \t\t{len(real_lines)} lines,\n   while we have \t{len(test_lines)} lines\n\033[0;0;0m")

    wrong_lines = [line not in real_lines for line in test_lines]
    missing_lines = [line not in test_lines for line in real_lines]

    wrong_lines_i = [(i if wrong_lines[i] else None) for i in range(len(wrong_lines))]
    missing_lines_i = [(i if missing_lines[i] else None) for i in range(len(missing_lines))]

    color = green if sum(wrong_lines) == 0 else red
    print(f"\033{color}Wrong lines   : {sum(wrong_lines)}\033[0;0;0m\t(present in output but not in solution)")
    print(f"\t{[i for i in wrong_lines_i if i is not None]}\n")

    color = green if sum(missing_lines) == 0 else red
    print(f"\033{color}Missing lines : {sum(missing_lines)}\033[0;0;0m\t(present in solution but missing in output)")
    print(f"\t{[i for i in missing_lines_i if i is not None]}\n")

    if del_output:
        os.system("del fs_test.tmp")

def test_basic_model_training(recompute=True, del_output=True):
    if recompute:
        return_value = os.system("python basic_model_training.py Datasets/molecules-small.pos Datasets/molecules-small.neg 5 5 4 > bmt_test.tmp")
        if return_value != 0:
            exit(return_value)
    
    with open('bmt_test.tmp', 'r') as test:
        test_lines = [line.strip() for line in test.readlines()]
    with open('Examples/task2_small_5_5_4.txt', 'r') as real:
        real_lines = [line.strip() for line in real.readlines()]

    red = "[1;31;1m"
    green = "[1;32;1m"

    color = green if len(real_lines) == len(test_lines) else red
    print(f"\033{color}Solution has \t\t{len(real_lines)} lines,\n   while we have \t{len(test_lines)} lines\n\033[0;0;0m")

    wrong_lines = [line not in real_lines for line in test_lines]
    missing_lines = [line not in test_lines for line in real_lines]

    wrong_lines_i = [(i if wrong_lines[i] else None) for i in range(len(wrong_lines))]
    missing_lines_i = [(i if missing_lines[i] else None) for i in range(len(missing_lines))]

    color = green if sum(wrong_lines) == 0 else red
    print(f"\033{color}Wrong lines   : {sum(wrong_lines)}\033[0;0;0m\t(present in output but not in solution)")
    print(f"\t{[i for i in wrong_lines_i if i is not None]}\n")

    color = green if sum(missing_lines) == 0 else red
    print(f"\033{color}Missing lines : {sum(missing_lines)}\033[0;0;0m\t(present in solution but missing in output)")
    print(f"\t{[i for i in missing_lines_i if i is not None]}\n")

    if del_output:
        os.system("del bmt_test.tmp")

if __name__ == '__main__':
    # test_finding_subgraph(recompute=True, del_output=True)
    test_basic_model_training(recompute=True, del_output=True)
