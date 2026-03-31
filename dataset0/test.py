def trim(s):
    if s == '':
        return s
    if ' ' in s[0]:
        s = s[1:]
        return trim(s)
    elif ' ' in s[-1]:
        s = s[:-1]
        return trim(s)
    return s

if trim('hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello') != 'hello':
    print('测试失败!')
elif trim('  hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello  world  ') != 'hello  world':
    print('测试失败!')
elif trim('') != '':
    print('测试失败!')
elif trim('    ') != '':
    print('测试失败!')
else:
    print('测试成功!')