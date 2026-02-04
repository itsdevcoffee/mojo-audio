# Mojo 0.26.2+ FFI Constraints - Battle-Tested Rules

**Version:** Mojo 0.26.2.0.dev2026020405 (Updated Feb 4, 2026)

**Status:** Production-validated through extensive debugging and testing

**Updates:** API changes for Origin syntax in Mojo 0.26.2+

---

## 🔥 Critical Constraints (DO NOT VIOLATE)

These were discovered through systematic testing and must be followed for FFI to work.

### 1. ❌ CANNOT Import Functions in Shared Libraries

```mojo
# BROKEN - causes segfault (exit 139):
from audio import mel_spectrogram
result = mel_spectrogram(...)

# WORKING - inline the implementation:
fn mel_spectrogram_ffi(...) raises -> List[List[Float32]]:
    # Complete implementation inlined (~650 lines)
```

**Reason:** Imported functions crash when called from shared library context

**Evidence:** Tested systematically - import works, but calling crashes

---

### 2. ❌ CANNOT Use `Int` for FFI Boundaries

```mojo
# BROKEN - ABI mismatch causes crashes:
fn get_size(handle: Int32) -> Int
fn process_data(buffer: UnsafePointer[...], size: Int)

# WORKING - use fixed-size types:
fn get_size(handle: Int32) -> UInt64  # For C's size_t
fn process_data(buffer: UnsafePointer[...], size: UInt64)
```

**Reason:** `Int` is platform-dependent (32 or 64 bit), causes ABI mismatch with C

**C Types Mapping:**
- `int32_t` → `Int32`
- `int64_t` → `Int64`
- `size_t` → `UInt64` (on 64-bit systems)
- `float` → `Float32`

---

### 3. ❌ CANNOT Return Structs by Value

```mojo
# BROKEN - immediate segfault:
@export("get_config", ABI="C")
fn get_config() -> MojoMelConfig:
    return MojoMelConfig()

# WORKING - use output pointer parameter:
@export("get_config", ABI="C")
fn get_config(out: UnsafePointer[mut=True, MojoMelConfig, MutAnyOrigin]):
    out[0] = MojoMelConfig()
```

**Reason:** Mojo 0.26.1 FFI doesn't support struct returns by value

**C Header Update Required:**
```c
// Change from:
MojoMelConfig mojo_mel_config_default(void);

// To:
void mojo_mel_config_default(MojoMelConfig* out_config);
```

---

### 4. ✅ MUST Use Handle-Based API for Complex Types

**Pattern:** Convert pointers to Int64 handles, avoid nested pointers

```mojo
# Handle conversion utilities:
@always_inline
fn _ptr_to_handle[T: AnyType](ptr: UnsafePointer[T]) -> Int64:
    return Int64(Int(ptr))

@always_inline
fn _handle_to_ptr(handle: Int64) -> UnsafePointer[mut=True, Type, MutAnyOrigin]:
    return UnsafePointer[mut=True, Type, MutAnyOrigin](
        unsafe_from_address=Int(handle)
    )
```

**Why:** Avoids nested pointer types which Mojo can't infer origins for

**C API:**
```c
typedef int64_t MojoMelHandle;  // NOT int32_t! (truncates 64-bit pointers)
```

---

### 5. ✅ MUST Specify Origin Parameters Explicitly

```mojo
# UnsafePointer v2 syntax (Mojo 0.26.2+):
UnsafePointer[mut=True, Type, MutAnyOrigin]        # For C input params (mutable)
UnsafePointer[mut=False, Type, ImmutAnyOrigin]     # For C input params (immutable)
UnsafePointer[mut=True, Type, MutAnyOrigin]        # For heap-allocated returns (mutable)
UnsafePointer[mut=False, Type, ImmutExternalOrigin]  # For string literals/constants
```

**API Changes in 0.26.2:**
- `MutOrigin.external` → `MutAnyOrigin` (for all mutable pointers)
- `ImmutOrigin.external` → `ImmutExternalOrigin` (for immutable external data)
- Origins are built-in comptime values - no import needed

**Constructor syntax:**
```mojo
# Use keyword argument for unsafe_from_address:
var ptr = UnsafePointer[mut=True, Int32, MutAnyOrigin](
    unsafe_from_address=address_value
)
```

---

### 6. ✅ MUST Use Correct @export Syntax

```mojo
@export("function_name", ABI="C")
fn function_name(...) -> ReturnType:
    # Must provide explicit name as string (first argument)
    # Must specify ABI="C" for C calling convention
    # Return type must be C-compatible
```

---

### 7. ✅ MUST Use Correct Import Paths and Function Names

```mojo
# Correct imports:
from memory import UnsafePointer
from memory.unsafe_pointer import alloc  # Function, not a type
from sys.info import size_of  # NOT sizeof!

# Allocation:
var ptr = alloc[Type](count)  # Use alloc function

# Initialization:
ptr.init_pointee_move(value)  # Move ownership
ptr.init_pointee_copy(value)  # Copy value

# Cleanup:
ptr.destroy_pointee()  # Call destructor
ptr.free()  # Free memory (method, not function!)
```

---

## 🚫 Common "Improvements" That Break FFI

| Suggestion | Why It Breaks |
|------------|---------------|
| "Import instead of inline" | Crashes - can't call imported funcs in shared lib |
| "Use `Int` instead of `UInt64`" | ABI mismatch with C's `size_t` |
| "Return struct by value" | Segfault - not supported |
| "Use Int32 for handles" | Truncates 64-bit pointers |
| "Remove origin parameters" | Compilation error |
| "Simplify to nested pointers" | Mojo can't infer origins |
| "Use `sizeof` instead of `size_of`" | `sizeof` doesn't exist |
| "Use `free(ptr)` function" | It's a method: `ptr.free()` |

---

## ✅ Safe Simplifications

**You MAY suggest:**
- Code formatting improvements
- Better comments/documentation
- Clearer variable names (within functions, not FFI boundary)
- Removing truly dead code
- Loop optimizations (if semantics identical)
- Consolidating redundant calculations

**You MAY NOT change:**
- Function signatures (FFI boundary)
- Type choices (Int32, Int64, UInt64)
- Inlined vs imported code
- Handle management logic
- Origin parameters
- Error handling patterns

---

## 🧪 Validation Before Changes

**Before suggesting any FFI-related change:**

1. ✅ Check Mojo version in `pixi.toml`
2. ✅ Search for current docs: "Mojo 0.26 [feature] 2026"
3. ✅ Verify change doesn't affect C ABI
4. ✅ Test builds: `pixi run build-ffi-optimized`
5. ✅ Test functionality: `LD_LIBRARY_PATH=. ./test_basic`

**If ANY test fails after change → revert immediately**

---

## 📚 Key Resources

- **Mojo version:** See `pixi.toml` lines 46-47
- **UnsafePointer docs:** https://docs.modular.com/mojo/manual/pointers/unsafe-pointers/
- **@export docs:** https://docs.modular.com/mojo/manual/decorators/export/
- **size_of docs:** https://docs.modular.com/mojo/stdlib/sys/info/size_of/
- **Working examples:** https://github.com/ihnorton/mojo-ffi

---

## 🎯 Test Matrix (All Must Pass)

After any changes:
- [ ] `pixi run build-ffi-optimized` succeeds
- [ ] `pixi run test` succeeds (core library)
- [ ] `gcc -I include -L. -lmojo_audio test_basic.c -o test_basic -lm` compiles
- [ ] `LD_LIBRARY_PATH=. ./test_basic` runs without crash
- [ ] Output: (80, 2998) frames for 30s audio

If any fail → **changes broke something critical**

---

**This document represents weeks of debugging Mojo 0.26.1 FFI. Respect these constraints.**
