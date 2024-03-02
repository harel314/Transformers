#%%
from transformers import pipeline,set_seed
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
#functions
def wrap(x):
    return textwrap.fill(x,replace_whitespace=False,fix_sentence_endings=True)

#%% upload generator
generator = pipeline('text-generation')
#%% upload the poem
poem = "poems/a_frog_and_a_toad.txt"
with open(poem) as f:
    poem_lines = f.readlines()
poem_lines = [line[:-2] for line in poem_lines]
#%% generate
pprint(generator(poem_lines[0]))
#%% control output 
pprint(generator(poem_lines[0],max_length=20))
pprint(generator(poem_lines[0],num_return_sequences = 3,max_length=25))

#%% write a poem
out = generator(poem_lines[0])
#%%
res = wrap(out[0]['generated_text'])
print(res)
pprint(generator(res))

