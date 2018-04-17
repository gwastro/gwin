# Documentation stucture

This document describes the structure of the documentation for GWIn

## How-to generate documentation

```
make html -j2
```

## ``/_includes``

The `/docs/_includes` directory is automatically parsed during `make html`,
allowing for on-the-fly generation of RST documentation.

Any Python files in there will be executed and the output dumped to an RST
file with the same base name, e.g. output from `_includes/foo.py` will be
written to `_includes/foo.rst`.

The output RST files can then be embedded into other RST files (in the
regular documentation) via RST `..include::` directives, e.g.:

```rst
.. include:: /_includes/foo.rst
```
