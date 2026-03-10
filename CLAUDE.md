# Claude Code Instructions - mojo-audio

## Documentation Organization

All markdown files go in `docs/` (except README.md, CLAUDE.md, LICENSE.md, CONTRIBUTING.md).
Never create ALL_CAPS.md files anywhere except those four special cases.

**Subdirectories and what belongs where:**

| Directory | What goes here | Date prefix? |
|-----------|---------------|-------------|
| `docs/context/` | Living reference: architecture, constraints, capabilities, cheat sheets | No |
| `docs/guides/` | Living tutorials and how-to docs for contributors / users | No |
| `docs/decisions/` | ADRs — why we made a specific technical choice | Yes |
| `docs/research/` | Point-in-time explorations, comparisons, experiments | Yes |
| `docs/plans/` | Implementation plans for specific features | Yes |
| `docs/handoff/` | Context documents for agent-to-agent or session handoffs | Yes |
| `docs/project/` | Roadmaps and high-level strategy | Yes |
| `docs/benchmarks/` | Benchmark results tied to a date/version | Yes |
| `docs/drafts/` | Work in progress, not yet published | No |

**Date prefix rule — the key distinction:**

- **Use a date prefix** when the document is a snapshot: a plan written at a point in time, a decision made, research done, results recorded. The date tells readers "this reflects what we knew on X date."
  - Example: `docs/plans/03-05-2026-hubert-max-engine-experiment.md`

- **No date prefix** when the document is living reference: a guide, a cheat sheet, an architecture overview. These should be updated in-place as things change. A date implies "snapshot" and misleads readers who expect current information.
  - Example: `docs/guides/ffi-integration-guide.md`

**Naming:** lowercase, hyphens for spaces, descriptive.
- With date: `MM-DD-YYYY-short-name.md`
- Without date: `short-descriptive-name.md`

**Before creating a new doc:** check if an existing one should be updated instead.

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
