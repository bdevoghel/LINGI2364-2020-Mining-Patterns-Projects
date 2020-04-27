import os

def test_correct_output(command:str, recompute, del_output, correction_file:str):
    if recompute:
            return_value = os.system(command  +  ' > test_output.tmp')
        if return_value != 0:
            exit(return_value)
    
    with open('test_output.tmp', 'r') as test:
        test_lines = [line.strip() for line in test.readlines()]
    with open(correction_file, 'r') as real:
        real_lines = [line.strip() for line in real.readlines()]

    red = "[1;31;1m"
    green = "[1;32;1m"

    print("\n" + correction_file + " : ")
    color = green if len(real_lines) == len(test_lines) else red
    print(f"\033{color}Solution has \t{len(real_lines)} lines,\n    ours has \t{len(test_lines)} lines\033[0;0;0m")

    wrong_lines = [line not in real_lines for line in test_lines]
    missing_lines = [line not in test_lines for line in real_lines]

    wrong_lines_i = [(i if wrong_lines[i] else None) for i in range(len(wrong_lines))]
    missing_lines_i = [(i if missing_lines[i] else None) for i in range(len(missing_lines))]

    color = green if sum(wrong_lines) == 0 else red
    print(f"\033{color}Wrong lines   : {sum(wrong_lines)}\033[0;0;0m\t(present in output but not in solution)")
    print(f"\t{[i for i in wrong_lines_i if i is not None]}")

    color = green if sum(missing_lines) == 0 else red
    print(f"\033{color}Missing lines : {sum(missing_lines)}\033[0;0;0m\t(present in solution but missing in output)")
    print(f"\t{[i for i in missing_lines_i if i is not None]}")

    if del_output:
        os.system("del test_output.tmp")


def test_finding_subgraph(recompute=True, del_output=True):
    test_correct_output("python finding_subgraphs.py Datasets/molecules-small.pos Datasets/molecules-small.neg 5 5", recompute, del_output, "Examples/task1_small_5_5.txt")

def test_basic_model_training(recompute=True, del_output=True):
    test_correct_output("python basic_model_training.py Datasets/molecules-small.pos Datasets/molecules-small.neg 5 5 4", recompute, del_output, "Examples/task2_small_5_5_4.txt")

def test_sequential_covering(recompute=True, del_output=True):
    test_correct_output("python sequential_covering.py Datasets/molecules-small.pos Datasets/molecules-small.neg 5 5 4", recompute, del_output, "Examples/task2_small_5_5_4.txt")


if __name__ == '__main__':
    # test_finding_subgraph(recompute=True, del_output=True)
    # test_basic_model_training(recompute=True, del_output=True)
    test_sequential_covering(recompute=True, del_output=True)
