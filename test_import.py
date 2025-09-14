import sys
sys.path.append(r'c:\Users\86136\Desktop\ben_01\Ben_Bot')
try:
    from plugins.Ben_learning_chat import active_chat
    print('Import successful!')
except Exception as e:
    print(f'Import failed: {e}')