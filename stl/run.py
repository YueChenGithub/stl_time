import os

# for script in ['1_ge', '2_ggee', '3_eegg', '4_gggg', '5_eeee']:
#     for method in ['0', 's', 't_min', 'c_min', 'c_sum2', 'G_s', 'G_c_min', 'mul', 'sum']:
#         for x0 in [0, 30]:
#             os.system(f"python {script}.py -method {method} -x0 {x0}")


for script in ['1_ge', '2_ggee', '3_eegg', '4_gggg', '5_eeee']:
    for method in ['xi']:
        for x0 in [0, 30]:
            os.system(f"python {script}.py -method {method} -x0 {x0}")




