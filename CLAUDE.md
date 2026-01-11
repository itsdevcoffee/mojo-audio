# Claude Code Instructions - mojo-audio

## Documentation Organization

All markdown files go in `docs/` (except README.md, CLAUDE.md, LICENSE.md, CONTRIBUTING.md).

**Subdirectories:**
- `docs/prompts/` - Prompts for copying into new agent sessions
- `docs/context/` - Context documents for future reference
- `docs/research/` - Research, comparisons, explorations
- `docs/project/` - High-level planning (todos, features, roadmap)
- `docs/architecture/` - System design, architectural patterns
- `docs/decisions/` - Design decisions, technical rationale (ADRs)
- `docs/guides/` - User-facing documentation and tutorials
- `docs/tmp/` - Temporary files (safe to delete anytime)

**File naming:** `[MM-DD-YYYY]-[short-name-with-hyphens].md`
- Example: `01-11-2026-mojo-ffi-constraints.md`
- Lowercase, hyphens for spaces
- Date prefix for chronological sorting

**Never create all-caps markdown files at project root** (except the special cases listed above).

## Mojo Language Specifics

**Version:** Check `pixi.toml` for current Mojo version before making assumptions about syntax.

**Documentation:** Mojo is rapidly evolving. Always verify syntax against current official docs:
- Use WebSearch with year: "Mojo [feature] 2026" or "Mojo 0.26 [feature]"
- Check: https://docs.modular.com/mojo/
- Reference: `docs/context/` for project-specific discoveries

**Experimental Language:** Mojo syntax changes frequently between versions. If unsure about a pattern:
1. Check pixi.toml for exact Mojo version
2. Search for version-specific documentation
3. Reference existing working code in the project
4. Ask before assuming syntax from outdated examples
