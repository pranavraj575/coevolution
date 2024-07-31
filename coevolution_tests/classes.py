class A:
    token = 'a'
    hypertoken = 'a'


def constructor(Class, token):
    class Subclass(Class):
        def __init__(self):
            super().__init__()
            self.token = token

    return Subclass


C = constructor(A, 'c')
print(C)
print(C().token)
print(C().hypertoken)