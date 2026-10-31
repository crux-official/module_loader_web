// The module loader driver for the web platform (Emscripten / wasm): the app
// is a side module, loaded at run time by the main module's dylink runtime
// through dlopen/dlsym/dlclose (the same names as POSIX; Emscripten's backend
// wires the shared linear memory AND the shared function table across the
// boundary, which is what lets the crux import-free dialect port directly -
// the both-direction dispatch was confirmed by the Phase 0 spike,
// .scratch/phase0).
//
// POSIX mapping semantics and the staging fix (mirroring module_loader_posix):
// dlopen caches a mapping by the path's name — a second dlopen of a path that
// is already mapped returns the existing handle (the bytes mapped the first
// time) without re-reading the file. A hot-reload swap of a rebuilt module at
// the same path would therefore keep running the boot-time image: the verify
// gate and the Migrate ask pass on the old code's own interface, the swap
// commits, and nothing changes on screen. The driver stages: at load time the
// source module is copied to a uniquely-named temp file in the same directory,
// and the staged copy — not the source — is what gets mapped. A fresh unique
// name per load means a fresh mapping per load, the source file is never
// mapped (a rebuild that replaces it is picked up), and the staged temp is
// deleted at unload.
//
// The source's backing differs by target: a node build has the side module as
// a file (the copy reads it through the filesystem), a browser build serves it
// over HTTP (the copy reads the fetched bytes the page has placed in the
// virtual FS). Either way the staged copy is a byte-for-byte twin and what the
// mapping is taken from.
//
// The per-load record holds the dlopen handle, both paths, and the source's
// last-write time (best-effort — a file-backed source has one, a URL-backed
// source does not), and lives in raw platform memory. sourcePath is the
// original module (what a change signal tracks); stagedPath is the temp the
// mapping was taken from (deleted at unload). Nothing composed today
// compares the mtime: the node/test trigger is an explicit frame
// count, and the browser's change fact is the page's own long-poll on
// the module's HTTP Last-Modified (any static server suffices — there
// is no dev server), which reads the header, not the FS mtime (the
// in-memory FS stamps the seed's write time, not the build time). The
// stat stays — the contract carries the field, and a reloader that
// lands here will use it.
#include "kernel_types.h"
#include "module_loader.h"
#include "module_loader_api.h"
#include "extension.h"
#include "memory.h"
#include "logging.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

// A per-process monotonic counter producing the unique part of each staged
// temp name. The engine ticks all modules from one thread; no synchronisation
// is needed.
static uint32_t g_staging_counter = 0;

// Unique staging name format: <sourcePath>.crux<n>.wasm
// Maximum suffix length: ".crux" + 10 digits (n) + ".wasm" = 5+10+5 =
// 20; the space is strictly greater than the suffix (the guard is
// >=) and the call sites add the +1 for the NUL themselves.
#define CRUX_STAGING_SUFFIX_SPACE 21

// The copy's chunk size: a multiple of the page size, large relative to the
// side modules this service is v1 for (tens of KB). The buffer is backed by
// the heap (the memory concern), not the stack — the web target's default
// stack (64KB) is the chunk size itself.
#define CRUX_COPY_CHUNK_SIZE 65536

typedef struct ModuleRecord {
  void *dlhandle;
  char *sourcePath; // The original module (what a change signal tracks)
  char *stagedPath; // The staged temp loaded from; deleted at unload
  int64_t mtime;    // Source last-write time (seconds); -1 if unavailable
} ModuleRecord;

// One staged copy's two ends. A named parameter pair: the two paths are both
// const char*, and a bare adjacent pair of them is what the repo's parameter
// checks flag.
typedef struct StageCopy {
  const char *source;
  const char *staged;
} StageCopy;

// Copy source to staged byte for byte. Any read or write failure leaves no
// partial staged file behind.
static bool StageFileCopy(const StageCopy *args) {
  FILE *src = fopen(args->source, "rb");
  if (src == NULL) {
    return false;
  }
  FILE *dst = fopen(args->staged, "wb");
  if (dst == NULL) {
    (void)fclose(src);
    remove(args->staged);
    return false;
  }
  // The chunk buffer is on the heap (not the stack): the web target's default
  // stack (64KB) is the chunk size itself, so a stack buffer would overflow on
  // the first load. The memory concern backs it (the package idiom); a failed
  // claim is a clean refusal (no partial temp).
  uint8_t *chunk = (uint8_t *) Memory_Alloc(CRUX_COPY_CHUNK_SIZE);
  size_t chunkBytes = 0;
  bool copySucceeded = (chunk != NULL);
  while (copySucceeded && feof(src) == 0) {
    chunkBytes = fread(chunk, 1, CRUX_COPY_CHUNK_SIZE, src);
    // A read can fail and still return a partial count (the error flag is
    // set, the position is indeterminate), so the state is checked before
    // the count: anything but a clean EOF stops the loop.
    if (ferror(src) != 0) {
      copySucceeded = false;
      break;
    }
    if (chunkBytes == 0) {
      break; // a clean EOF: the copy is complete
    }
    if (fwrite(chunk, 1, chunkBytes, dst) != chunkBytes) {
      copySucceeded = false;
      break;
    }
  }
  (void)fclose(src);
  const int closeResult = fclose(dst);
  Memory_Free(chunk);
  if (!copySucceeded || closeResult != 0) {
    remove(args->staged); // no partial temp may survive a failed copy
    return false;
  }
  return true;
}

// The staged temp lives in the same directory as the source: the build's
// output directory, which is writable. Returns a new allocation (caller
// frees) or NULL on failure.
static char *BuildStagingPath(const char *sourcePath) {
  const size_t pathLen = strlen(sourcePath);
  char *staged = (char *)Memory_Alloc(pathLen + CRUX_STAGING_SUFFIX_SPACE + 1);
  if (staged == NULL) {
    return NULL;
  }
  const uint32_t counter = g_staging_counter++;
  const int nWritten = snprintf(staged, pathLen + CRUX_STAGING_SUFFIX_SPACE + 1,
                                "%s.crux%u.wasm", sourcePath, (unsigned)counter);
  if (nWritten < 0 || (size_t)nWritten >= pathLen + CRUX_STAGING_SUFFIX_SPACE) {
    Memory_Free(staged);
    return NULL;
  }
  return staged;
}

void *Module_Load(const char *path) {
  if (path == NULL) {
    // dlopen accepts NULL (the global scope); the contract's path does not.
    return NULL;
  }

  // 1. Build the staged temp path. A unique name per load is what makes the
  //    mapping fresh: an already-mapped name would hand back the old bytes.
  char *stagedBuf = BuildStagingPath(path);
  if (stagedBuf == NULL) {
    return NULL;
  }

  // 2. Copy the source to the staged location. The staged copy's mtime is an
  //    artifact of the copy and is never used for any comparison.
  {
    const StageCopy copy = { .source = path, .staged = stagedBuf };
    if (!StageFileCopy(&copy)) {
      LogError("module_loader_web: staging copy of '%s' failed.", path);
      Memory_Free(stagedBuf);
      return NULL;
    }
  }

  // 3. Map the staged copy (not the source). The side module imports nothing
  //    from the host (the crux import-free dialect), so its imports resolve
  //    at load time (RTLD_NOW) and its symbols stay out of the global scope
  //    (RTLD_LOCAL).
  void *dlhandle = dlopen(stagedBuf, RTLD_NOW | RTLD_LOCAL);
  if (dlhandle == NULL) {
    // The host reports the failure by the source path; the reason (a
    // truncated read of a mid-write source, a missing dependency) comes from
    // the loader itself.
    LogError("module_loader_web: mapping '%s' failed: %s", stagedBuf, dlerror());
    remove(stagedBuf);
    Memory_Free(stagedBuf);
    return NULL;
  }

  // 4. Build the record.
  ModuleRecord *record = (ModuleRecord *)Memory_Alloc(sizeof(ModuleRecord));
  const size_t pathSize = strlen(path) + 1;
  const size_t stagedSize = strlen(stagedBuf) + 1;
  char *sourceCopy = (char *)Memory_Alloc(pathSize);
  char *stagedCopy = (char *)Memory_Alloc(stagedSize);
  if (record == NULL || sourceCopy == NULL || stagedCopy == NULL) {
    dlclose(dlhandle);
    remove(stagedBuf);
    Memory_Free(record);
    Memory_Free(sourceCopy);
    Memory_Free(stagedCopy);
    Memory_Free(stagedBuf);
    return NULL;
  }
  memcpy(sourceCopy, path, pathSize);
  memcpy(stagedCopy, stagedBuf, stagedSize);
  Memory_Free(stagedBuf); // The record's copy is now authoritative
  record->dlhandle = dlhandle;
  record->sourcePath = sourceCopy;
  record->stagedPath = stagedCopy;
  record->mtime = -1;

  // 5. Best-effort last-write time (the mtime the driver returns via
  //    Module_GetMtime). A file-backed source (node) has one; a URL-backed
  //    source (browser) does not — the stat fails and the record keeps -1.
  struct stat fileInfo;
  if (stat(path, &fileInfo) == 0) {
    record->mtime = (int64_t)fileInfo.st_mtime;
  }
  return record;
}

void Module_Unload(void *handle) {
  if (handle == NULL) {
    return;
  }
  ModuleRecord *record = (ModuleRecord *)handle;
  // The dlopen handle is invalid after dlclose, so it is released before
  // the record holding it is freed. The staged temp (never the source) is
  // the file the mapping was taken from, so the name goes with the mapping.
  dlclose(record->dlhandle);
  remove(record->stagedPath);
  Memory_Free(record->sourcePath);
  Memory_Free(record->stagedPath);
  Memory_Free(record);
}

void *Module_GetSymbol(void *handle, const char *name) {
  if (handle == NULL || name == NULL) {
    return NULL;
  }
  ModuleRecord *record = (ModuleRecord *)handle;
  return dlsym(record->dlhandle, name);
}

bool Module_GetMtime(void *handle, int64_t *outMtimeSeconds) {
  if (handle == NULL || outMtimeSeconds == NULL) {
    return false;
  }
  ModuleRecord *record = (ModuleRecord *)handle;
  if (record->mtime < 0) {
    return false;
  }
  *outMtimeSeconds = record->mtime;
  return true;
}

bool Module_FileName(const char *baseName, char *out, size_t size) {
  if (baseName == NULL || out == NULL) {
    return false;
  }
  size_t prefixSize = strlen(CMAKE_SHARED_LIBRARY_PREFIX);
  size_t baseSize = strlen(baseName);
  size_t suffixSize = strlen(CMAKE_SHARED_LIBRARY_SUFFIX);
  // prefix + baseName + suffix + NUL: the build's own SHARED target naming
  // (on web the suffix is .wasm, the side module's real file extension).
  if (size < prefixSize + baseSize + suffixSize + 1) {
    return false;
  }
  size_t offset = 0;
  memcpy(out, CMAKE_SHARED_LIBRARY_PREFIX, prefixSize);
  offset += prefixSize;
  memcpy(out + offset, baseName, baseSize);
  offset += baseSize;
  memcpy(out + offset, CMAKE_SHARED_LIBRARY_SUFFIX, suffixSize);
  offset += suffixSize;
  out[offset] = '\0';
  return true;
}

// The served struct: the module concern's contract, filled with this
// driver's function addresses. Valid from load, before Construct.
static ModuleAPI g_module_api = {
    .version = {2, 0, 0},
    .Load = Module_Load,
    .Unload = Module_Unload,
    .GetSymbol = Module_GetSymbol,
    .GetMtime = Module_GetMtime,
    .FileName = Module_FileName,
};

static bool module_loader_web_construct(void) {
  Engine_RegisterConcern(CRUX_CONCERN_MODULE);
  return true;
}

static void *module_loader_web_get_api(void) {
  return &g_module_api;
}

ExtensionInterface g_extension_module_loader_web = {
    "module_loader_web",
    {1, 0, 0},
    module_loader_web_construct,
    NULL,
    NULL,
    NULL,
    NULL,
    module_loader_web_get_api,
};
