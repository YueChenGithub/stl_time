import os

# sanity check
# for j in ['ggee_0', 'gggg_0',  'ggggg_0', 'eeee_0',  'eeeee_0']:
#     assert os.path.isfile(f'{j}.py')
# for j in ['ggee_1', 'gggg_1', 'ggggg_1']:
#     assert os.path.isfile(f'{j}.py')
# for j in ['ge_2', 'ggee_2', 'gggg_2',  'ggggg_2', 'eeee_2',  'eeeee_2']:
#     assert os.path.isfile(f'{j}.py')

# for r in ['s', 't_min', 't_sum', 't_left', 't_right', 'c_min']:
#     for j in ['ge_0', 'ge_1']:
#         os.system(f"python {j}.py -robustness {r}")

for j in ['ggee_0', 'gggg_0',  'ggggg_0', 'eeee_0',  'eeeee_0']:
    for r in ['s', 't_min', 'c_min']:
        os.system(f"python {j}.py -robustness {r}")

for j in ['ggee_1', 'gggg_1', 'ggggg_1']:
    for r in ['s', 'c_min']:
        os.system(f"python {j}.py -robustness {r}")

for j in ['ge_2', 'ggee_2', 'gggg_2',  'ggggg_2', 'eeee_2',  'eeeee_2']:
    for r in ['mul', 'sum']:
        os.system(f"python {j}.py -robustness {r}")