from numpydoc.docscrape import NumpyDocString
from glob import glob

pipeline_steps = glob('../autoprof/pipeline_steps/*.py')

param_docs = {}

for step in pipeline_steps:
    func_docs = {}
    print(step)
    count_funcs = 0
    with open(step, 'r') as f:
        in_func = False
        in_doc = False
        func = ''
        for l in f.readlines():
            if l[:3] == 'def':
                in_func = True
                count_funcs += 1
                func = l[3:l.find('(')].strip()
                func_docs[func] = ''
                continue
                
            if in_doc and ('"""' in l):
                in_doc = False
                in_func = False
                #print(func_docs[func])
                continue
            if in_doc:
                func_docs[func] = func_docs[func] + l
            if not in_doc and '"""' in l:
                in_doc = True
                func_docs[func] = func_docs[func] + '    ' + l.strip(' "\n\t') + '\n\n'
    print('N funcs: ', count_funcs)
    for func in func_docs:
        if func[0] == '_':
            continue
        docstring = NumpyDocString(func_docs[func])
        for P in docstring['Parameters']:
            if P.name not in param_docs:
                param_docs[P.name] = [{'parameter': P, 'func': func, 'step': step}]
            else:
                param_docs[P.name].append({'parameter': P, 'func': func, 'step': step})

parameters = ['='*20 + '\n', 'AutoProf Parameters\n', '='*20 + '\n\n',
              "Here we list all parameters used in the built-in AutoProf methods. The parameters are listed alphabetically for easy searching. For further information, links are included to the individual methods which use these parameters.\n\n"]

for name in sorted(param_docs.keys()):
    parameters.append(f"{name} ({param_docs[name][0]['parameter'].type})\n")
    parameters.append('-'*70 + '\n\n')
    parameters.append('**Referencing Methods**\n\n')
    for P in param_docs[name]:
        step = P['step'][P['step'].rfind('/')+1:-3]
        parameters.append(f"- :func:`~autoprof.pipeline_steps.{step}.{P['func']}`\n")
    parameters.append('\n')
    parameters.append('**Description**\n\n')
    for d in param_docs[name][0]['parameter'].desc:
        parameters.append(d + '\n')
    parameters.append('\n')

with open('parameters.rst', 'w') as f:
    f.writelines(parameters)
    
    


