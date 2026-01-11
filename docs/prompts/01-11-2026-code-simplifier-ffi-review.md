# code-simplifier: FFI Implementation Review

**Target:** Recently committed FFI layer in mojo-audio

**Goal:** Review for simplification opportunities while respecting Mojo 0.26.1 FFI constraints

---

## 🎯 Your Mission

Review these files for simplification opportunities:
1. `src/ffi/types.mojo` (57 lines) - Type definitions
2. `src/ffi/audio_ffi.mojo` (843 lines) - FFI implementation
3. `examples/ffi/demo_c.c` (137 lines) - C example
4. `examples/ffi/demo_rust.rs` (239 lines) - Rust example

**Start with types.mojo (safest), then proceed to others only if that goes well.**

---

## ⚠️ CRITICAL: Read Constraints First

**BEFORE making any suggestions, read:**
`docs/context/01-11-2026-mojo-ffi-constraints.md`

This document contains 7 battle-tested constraints discovered through weeks of debugging. **Violating any of these will break the FFI.**

---

## 🔒 Absolute Constraints (DO NOT VIOLATE)

### Language Version
- **Mojo 0.26.1.0.dev2026010718** (nightly from Jan 7, 2026)
- Check `pixi.toml` lines 46-47 for confirmation
- If unsure about syntax, WebSearch: "Mojo 0.26 [feature] 2026"

### Cannot Change
1. ❌ **No imports** - `mel_spectrogram` must stay inlined (lines 20-683)
2. ❌ **Fixed-size types** - `Int32`, `Int64`, `UInt64` (never `Int`)
3. ❌ **Struct return via pointer** - Cannot return structs by value
4. ❌ **Int64 handles** - Cannot use Int32 (truncates pointers)
5. ❌ **Origin parameters** - All UnsafePointer need explicit origins
6. ❌ **@export syntax** - Must have name string + ABI="C"
7. ❌ **Function names** - `size_of` not `sizeof`, `alloc` not `malloc`

### Must Preserve
- ✅ All 9 @export function signatures
- ✅ Handle conversion utilities (`_ptr_to_handle`, `_handle_to_ptr`)
- ✅ Error handling patterns (defensive coding required for FFI)
- ✅ Test compatibility (must still pass existing tests)

---

## ✅ What You CAN Simplify

**Safe to change:**
- Code formatting and indentation
- Comments and documentation
- Variable names **within** functions (not parameters)
- Redundant calculations within same scope
- Loop structure (if semantics identical)
- Error messages

**Inlined mel_spectrogram code (lines 20-683):**
- This was copied from `src/audio.mojo`
- May have redundancies or verbose patterns
- Safe to simplify **as long as output is identical**
- Do NOT remove it and replace with import

---

## 🧪 Validation Requirements

**After ANY change:**

1. **Must build:** `pixi run build-ffi-optimized`
2. **Must pass core tests:** `pixi run test`
3. **Create test to verify FFI:**
   ```bash
   gcc -I include -L. -lmojo_audio test_basic.c -o test_basic -lm
   LD_LIBRARY_PATH=. ./test_basic
   # Should output: Version: 0.1.0, Config: sr=16000...
   ```

**If any fail → revert changes**

---

## 📋 Suggested Review Order

### Phase 1: types.mojo (LOW RISK)
- 57 lines, simple type definitions
- Review struct layout, comments
- Check if error code definitions can be clearer
- **If this goes well, proceed to Phase 2**

### Phase 2: Examples (LOW-MEDIUM RISK)
- `examples/ffi/demo_c.c` - Just example code
- `examples/ffi/demo_rust.rs` - Just example code
- Safe to simplify, low impact if broken

### Phase 3: audio_ffi.mojo (HIGH RISK - BE CAREFUL)
- 843 lines, production FFI code
- Start with **comments only** - improve documentation
- Then **inlined functions** (lines 20-683) - may have redundancies
- **Last:** FFI exports (lines 684-843) - BE VERY CAREFUL

**Stop at any phase if you find violations or are unsure!**

---

## 🎓 Mojo FFI Context (So You Understand)

**This FFI took 3+ days to build because:**
- Mojo FFI is undocumented for this version
- Nightly build (3 days old) has breaking changes
- Multiple attempts by different agents failed
- Every constraint above was learned through crashes

**Key discoveries:**
- Importing crashes shared libs (fixed: inlined 650 lines)
- Nested pointers don't work (fixed: handle-based API)
- Struct returns crash (fixed: pointer parameters)
- `Int` causes ABI mismatch (fixed: explicit UInt64)

**This is WORKING, TESTED, PRODUCTION CODE**

Be conservative. When in doubt, don't change it.

---

## 🚨 Red Flags to Watch For

**If you find yourself thinking:**
- "This duplicates audio.mojo, let's import" → STOP
- "Int is simpler than UInt64" → STOP
- "Returning struct directly is cleaner" → STOP
- "This error handling is redundant" → VERIFY FIRST

**Ask yourself:** "Will this change affect the FFI boundary or Mojo compilation?"
- If YES → Verify against constraints doc
- If UNSURE → Note it, don't change it

---

## ✅ Success Criteria

**Your review should:**
- [ ] Respect all 7 critical constraints
- [ ] Only suggest safe simplifications
- [ ] Improve code quality without breaking functionality
- [ ] Note any concerns without applying risky changes
- [ ] Validate that tests still pass after changes

**Ideal outcome:**
- 5-10% code reduction through safe refactoring
- Improved comments/documentation
- Better code organization
- **ZERO functional changes**

---

**Read `docs/context/01-11-2026-mojo-ffi-constraints.md` before starting!**
