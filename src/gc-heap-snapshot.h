#ifndef JL_GC_HEAP_SNAPSHOT_H
#define JL_GC_HEAP_SNAPSHOT_H

#include "julia.h"

#ifdef __cplusplus
extern "C" {
#endif

struct HeapSnapshot;

void serialize_heap_snapshot(JL_STREAM *stream, struct HeapSnapshot *snapshot);

// ---------------------------------------------------------------------
// Functions to call from GC when heap snapshot is enabled
// ---------------------------------------------------------------------
// TODO: remove JL_DLLEXPORT
JL_DLLEXPORT void record_edge_to_gc_snapshot(jl_value_t *a, jl_value_t *b);

// ---------------------------------------------------------------------
// Functions to call from Julia to start heap snapshot
// ---------------------------------------------------------------------
// ...
JL_DLLEXPORT void take_gc_snapshot(void);


#ifdef __cplusplus
}
#endif


#endif  // JL_GC_HEAP_SNAPSHOT_H
