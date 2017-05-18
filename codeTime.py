#!python3
"""
codeTime.py
"""
import timeit


def exeTime(method):
    def timed(*args, **kw):
        ts = timeit.default_timer()
        result = method(*args, **kw)
        te = timeit.default_timer()

        print(
            '%r (%r, %r) %2.8f sec' % \
            (method.__name__, args, kw, te - ts)
        )
        return result

    return timed
