import ast

# 读取文件内容
try:
    with open('active_chat.py', 'r', encoding='utf-8') as f:
        code = f.read()
        
    # 解析语法树
    ast.parse(code)
    print('Syntax check passed! No syntax errors found.')
except SyntaxError as e:
    print(f'Syntax error found: {e}')
except Exception as e:
    print(f'Error reading or parsing file: {e}')