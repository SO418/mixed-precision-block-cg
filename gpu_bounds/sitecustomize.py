# Replace CuPy's hardcoded -ftz=true so subnormals are preserved (gradual underflow).
try:
    import cupy.cuda.compiler as _c

    _orig = _c._NVRTCProgram.compile

    def _compile(self, options=(), log_stream=None):
        options = tuple("-ftz=false" if o in ("-ftz=true", "--ftz=true") else o for o in options)
        return _orig(self, options, log_stream)

    _c._NVRTCProgram.compile = _compile
except Exception:
    pass
