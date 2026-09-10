"""Compile generated, typed Python functions for scene-dependent dispatch.

Only OrdinaryShade's restricted frontend lowers the function bodies. Python
execution defines their signatures; it never executes a shader body. The source
cache exists just for inspection during compilation, under a reentrant lock.
"""
import hashlib
import linecache
import threading
import ordinaryshade as osh

_lock = threading.RLock()


def compile_typed(source, name, *, namespace=None, externals=(), values=None):
    filename = '<ordinarylight-shader-' + hashlib.sha256(source.encode()).hexdigest() + '>'
    with _lock:
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        try:
            scope = {'osh': osh, **(namespace or {})}
            exec(compile(source, filename, 'exec'), scope)
            return osh.compile_function(osh.function(scope[name]), externals=externals,
                                        external_values={**{key: value for key, value in (namespace or {}).items() if isinstance(value, osh.StructType)}, **(values or {})}).source
        finally:
            linecache.cache.pop(filename, None)


def external_signature(name, parameters, result, namespace=None):
    if not name.isidentifier() or not name.isascii():
        raise ValueError('Shader function names must be ASCII identifiers')
    scope = {'osh': osh, **(namespace or {})}
    exec(f'def {name}({parameters}) -> {result}:\n    pass\n', scope)
    return osh.external(scope[name])


def lookup_source(name, parameter, pairs, default, *, boolean=False):
    """Specialize a typed lookup over immutable scene metadata."""
    result_type = 'osh.boolean' if boolean else 'osh.u32'
    literal = repr if boolean else lambda value: f'osh.u32({int(value)})'
    body = f'def {name}({parameter}: osh.u32) -> {result_type}:\n'
    for key, value in pairs:
        body += f'    if {parameter} == osh.u32({int(key)}):\n        return {literal(value)}\n'
    body += f'    return {literal(default)}\n'
    return compile_typed(body, name)
