"""Defense in depth against accidental Python writes outside the isolated app.

This is an accidental-write guard, not an OS security sandbox for native code.
Only audited read-only research functions are invoked by the adapters.
"""
import os
import sys
from pathlib import Path

def install(app_directory):
    allowed = app_directory.resolve()
    def check(value):
        if isinstance(value, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(value)).resolve()
            if not path.is_relative_to(allowed):
                raise PermissionError('Research protection blocked a write outside desktop_app: '+str(path))
    def guard(event, args):
        if event == 'open':
            path, mode, flags = args
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (flags and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
                check(path)
        elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime','os.truncate'}:
            check(args[0])
        elif event in {'os.rename','os.link','os.symlink'}:
            check(args[0]); check(args[1])
    sys.addaudithook(guard)
