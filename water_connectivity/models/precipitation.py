def const_precipitation(value=0.5):
    def func(_):
        return value

    return func
