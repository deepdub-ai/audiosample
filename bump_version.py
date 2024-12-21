# bump version in audiosample/__init__.py and setup.py using fire
import fire
import re

def bump_version(new_version):
    # check if new_version is a valid version
    if not re.match(r'^\d+\.\d+\.\d+$', new_version):
        raise ValueError("Invalid version format. Please use X.Y.Z format.")

    with open('audiosample/__init__.py', 'r') as file:
        lines = file.readlines()

    with open('setup.py', 'r') as file:
        lines = file.readlines()
    # Update version in __init__.py
    with open('audiosample/__init__.py', 'r') as file:
        init_lines = file.readlines()
    
    for i, line in enumerate(init_lines):
        if line.startswith('__version__'):
            init_lines[i] = f'__version__ = "{new_version}"\n'
            break
    
    with open('audiosample/__init__.py', 'w') as file:
        file.writelines(init_lines)

    # Update version in setup.py
    with open('setup.py', 'r') as file:
        setup_lines = file.readlines()
    
    for i, line in enumerate(setup_lines):
        if line.strip().startswith('version='):
            setup_lines[i] = f'    version="{new_version}",\n'
            break
    
    with open('setup.py', 'w') as file:
        file.writelines(setup_lines)

if __name__ == '__main__':
    fire.Fire(bump_version)