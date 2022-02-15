import os

inp_base = 'match_ID_800d2_SP'

inp_str = f'{inp_base}.ipynb'


cmd = f'jupyter nbconvert --to script {inp_str} --stdout > {inp_base}.py'

os.system(cmd)

# Run validation from command line via
# ipython <script>.py
