from numpydoc.docscrape import NumpyDocString
from glob import glob

pipeline_steps = glob('../autoprof/pipeline_steps/*.py')

func_docs = {}

for step in pipeline_steps:
    with open(step, 'r') as f:
        in_func = False
        in_doc = False
        func = ''
        for l in f.readlines():
            if not in_func and l[:3] == 'def':
                in_func = True
                func = l[3:l.find('(')].strip()
                func_docs[func] = ''
                continue
                
            if in_doc and ('"""' in l):
                in_doc = False
                in_func = False
                print(func_docs[func])
                break
            #continue
            if in_doc:
                func_docs[func] = func_docs[func] + l
            if not in_doc and '"""' in l:
                in_doc = True
                func_docs[func] = func_docs[func] + '    ' + l.strip(' "\n\t') + '\n\n'
        
param_docs = {}

for func in func_docs:
    docstring = NumpyDocString(func_docs[func])
    print(func)
    for key in docstring.keys():
        print(key, docstring[key])
    break
