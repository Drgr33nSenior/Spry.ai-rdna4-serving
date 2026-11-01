# Standard loaders for source fingerprints

Six source-fingerprint sites used direct `compile` calls: three in the worker,
two in the head observer, and one in the retained verify adapter. They now use
`SourceFileLoader.source_to_code` on the same source bytes and filename. This
API returns a code object without importing or executing the source and without
consulting cached bytecode.

The fingerprint comparisons, callable checks, source hashes and failure paths
remain in place. The new tests evaluate the actual six call expressions and
compare their code objects with the previous compilation result, including
nested factories and future flags. An unchecked stale bytecode file cannot
replace that result. The actual adapter method check still rejects a replacement
that borrows the authentic filename.

The verify adapter is now a canonical repository source. It was copied only after
all 313 donor members authenticated. Its original donor hash was added to the
source registry; existing original hashes remain unchanged. The package still
contains 75 review members and fits the 80-member archive ceiling. The canonical
source set grows from 20 to 21 files.

The focused checks pass on Python 3.11.6 and Python 3.14.8. The full repository
suite passes on macOS Python 3.14.8: 345 cleanup tests and 782 Python unit tests.
The complete proposed composition passes 35 checks on that interpreter, including
both serving replays and its package-wide AST check. Every proposed Python source
has no direct `exec` or `compile` call.

The effective proposed package includes the previously isolated protected
transport loader change. The canonical builder still emits the original nine
protected helpers, pending exact review. Their proposed bytes and hash set do not
change in this increment. Linux/native execution, production mount isolation and
the other cleanup gates remain open. No GPU work or inference performance
measurement was performed.
