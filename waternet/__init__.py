import sys
import os

# print('Importing waternet package')

# Install required packages not originally considered in MSPC
try:
    import patchify
except Exception as e:
    print('Installing patchify')
    stream = os.popen('pip install patchify --quiet')
    print(stream.read())

try:
    import fastai
except Exception as e:
    print('Fastai is needed to use waternet package')
    # stream = os.popen('mamba install -q -y -c fastchan fastai')
    # print(stream.read())
    
