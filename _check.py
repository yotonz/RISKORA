import sys
try:
    with open('app.py', 'r', encoding='utf-8') as f:
        src = f.read()
    compile(src, 'app.py', 'exec')
    with open('_check_result.txt', 'w') as out:
        out.write('SYNTAX OK\n')
except SyntaxError as e:
    with open('_check_result.txt', 'w') as out:
        out.write(f'SyntaxError line {e.lineno}: {e.msg}\n')
        out.write(f'Text: {repr(e.text)}\n')
except Exception as e:
    with open('_check_result.txt', 'w') as out:
        out.write(f'Error: {e}\n')
