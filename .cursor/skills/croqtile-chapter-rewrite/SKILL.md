---
name: croqtile-chapter-rewrite
description: >-
  Rewrites Croqtile tutorial chapters (ch01–ch09) to match the storytelling,
  figure, and structure standards established in ch02 and ch03. Use when
  rewriting, reviewing, or restructuring any tutorial chapter, or when the
  user mentions chapter quality, chapter structure, or tutorial consistency.
---

# Croqtile Tutorial Chapter Rewrite Standard

This skill encodes the writing and structural conventions that emerged from
human-reviewed iterations of Chapters 2 and 3 of the Croqtile tutorial.
Apply these rules when rewriting or reviewing any tutorial chapter.

## 1. Opening Pattern (Mandatory)

Every chapter opening has **two beats** in order:

### Beat 1 — Recap + Bridge

Start by referencing what the reader already knows from the *previous* chapter.
Use a concrete detail, not a vague "in the last chapter we learned…"

> **Good** (ch03): "The previous two chapters kept things simple: `parallel {i, j}
> by [4, 8]` created 32 instances, each handling one element, and we never asked
> where those instances actually ran."

> **Good** (ch02): "Chapter 1 expressed computation at the level of individual
> elements: pick position `(i, j)`, read the two inputs, add, write the result."

Then bridge into **why that is insufficient** for the topic this chapter covers.
The bridge is a single narrative sentence or short paragraph — not a bullet list.

### Beat 2 — Concept Introduction (Storytelling)

Before showing any Croqtile syntax, explain the **concept itself** using
general knowledge the reader can relate to:

- What is this concept/model/mindset in the real world or in GPU hardware?
- How do traditional tools (CUDA, OpenCL) handle it, and what makes it hard?
- What is Croqtile's key insight or design choice for this concept?

This section reads like a mini-essay — conversational, specific, opinionated.
Avoid bullet-point walls. Use analogies, contrasts, and concrete numbers.

> **Pattern** (ch02): "A GPU does not fetch one 32-bit integer from memory at a
> time. It fetches contiguous blocks — 128 bytes, 256 bytes — in a single
> transaction… Croqtile is designed around this insight."

> **Pattern** (ch03): "Parallelism is a virtual concept… CUDA conflates logical
> structure and physical mapping… Croqtile untangles them."

## 2. Single Storyline per Chapter

Each chapter has **one main storyline** expressed in its title. Every section
must advance that storyline. If a section could stand alone without the
surrounding context, it probably belongs in a different chapter or should be
absorbed into a larger section.

### Main vs Supporting Concepts

When a concept **supports** the main storyline but is not itself the storyline,
handle it as a subsection or a brief aside — not a top-level section:

| Treatment | When to use | Examples |
|-----------|-------------|---------|
| Subsection within a larger section | Concept is needed to understand the main topic | Memory specifiers in ch02, shared reuse in ch03 |
| "Aside" box (## heading + short explanation) | Concept is tangential but referenced later | Type system aside (mdspan/ituple) in ch03 |
| Cross-reference only | Concept is documented elsewhere in detail | "See the [mdspan reference](…) for full details" |
| Move to another chapter | Concept belongs to a different storyline | Move matmul out of a "data movement" chapter |

### Detecting Misplaced Content

Ask these questions for every `##` section:

1. Does this section **directly advance** the chapter's title topic?
2. If I removed this section, would the chapter's story break?
3. Could a reader understand this section without having read the sections before it in *this* chapter?

If the answer to (1) is no and (3) is yes, the section is a candidate for
relocation or absorption.

## 3. Figure Policy

### When to Add a Figure

A figure is warranted when:

- A **spatial relationship** exists (memory layout, hardware hierarchy, tiling grid)
- A **before/after** or **comparison** would clarify (per-element vs block, local vs shared)
- The concept involves **mapping** between two systems (code → hardware)
- The section is **> 300 words** of dense explanation without visual relief

A figure is NOT warranted when:

- The content is a short syntax reference (table is better)
- The code example is self-explanatory
- The section is already short (< 150 words)

### Figure Conventions

- Use Manim via `figures/theme.py` for dark/light variants
- Name: `ch0X_figN_short_name.py`
- Output: `docs/assets/images/ch0X/figN_short_name_{dark,light}.png`
- In markdown, always include both variants:
  ```
  ![Alt text](../assets/images/ch0X/figN_name_dark.png#only-dark)
  ![Alt text](../assets/images/ch0X/figN_name_light.png#only-light)
  ```
- Add a one-line italic caption below
- Add the new figure script to `figures/render_all.sh`

### Figure Placement

Place figures **immediately after** the paragraph that introduces the concept,
before the detailed explanation. The figure gives the reader a mental model,
then the text fills in the details. This is the ch02/ch03 pattern.

## 4. Section Granularity

### Avoid Over-Fragmentation

Do NOT create a `##` heading for every minor concept. Small related concepts
should be explained inline within a larger section. The rule of thumb:

- If a section has < 100 words of content, it should be merged into its parent
- If 3+ consecutive sections each discuss one small syntax form, combine them
  under a single heading like "Core Syntax" or "Key Operations"

### Avoid Under-Explanation

Do NOT create a `##` heading and then put only 1–2 sentences under it.
Either expand the content to justify the heading, or remove the heading and
inline the content.

### The Detail Test

For each subsection, ask: "Is this level of detail appropriate for a *tutorial*,
or does it belong in a *reference* document?"

- Tutorial: explain the *why* and *when* with one clear example
- Reference: exhaustive listing of every option, edge case, and variant

If you find yourself listing every flag, every overload, or every edge case,
move that content to a reference doc and cross-reference it.

## 5. Code Examples

### Incremental Composition

Code examples must **only use syntax and concepts introduced in the current or
previous chapters**. Never use syntax that has not been explained yet.

If an example needs a concept from a later chapter, either:
- Simplify the example to avoid it
- Add a brief forward-reference: "We use `X` here; it is explained in Chapter N"

### Code-Text Integration

After each code block, explain it **piece by piece**, using the exact variable
names and expressions from the code. The pattern from ch02/ch03:

1. Show the full code block
2. Immediately below, use **bold inline headers** to walk through each notable line:

> **Output shape from operand dimensions.** `s32 [lhs.span(0), rhs.span(1)] output`
> builds the output shape from the inputs…

> **Multi-axis parallel.** `parallel p by 16, q by 64` declares two parallel indices…

Do NOT put the explanation in code comments. The text IS the explanation.

### Host Code

Keep host code minimal and boring. Show it once per chapter (or skip it if
identical to previous chapters). The `__co__` function is the star.

## 6. Tone and Style

- Conversational but precise — like a knowledgeable colleague explaining at a whiteboard
- Use "you" to address the reader directly
- Avoid marketing language ("powerful", "elegant", "seamlessly")
- Avoid AI-generated patterns: no "Let's dive in", no "In this section we will explore"
- Use contractions naturally (it's, don't, you'll)
- Sentences should be varied in length — mix short declarative with longer explanatory
- Each paragraph should have one clear point; if it has two, split it

## 7. Chapter-End Pattern

Every chapter ends with:

1. **Summary table** — new syntax introduced, in a `| Syntax | Meaning |` table
2. **Bridge to next chapter** — one paragraph that says what comes next and why
   it builds on what was just learned. Be specific: name the concepts.

## 8. Rewrite Checklist

When rewriting a chapter, execute these steps in order:

- [ ] Read the existing chapter fully
- [ ] Identify the single main storyline (from the chapter title)
- [ ] Check each `##` section: does it belong? should it merge/move/absorb?
- [ ] Write the opening (Beat 1: recap+bridge, Beat 2: concept intro)
- [ ] For each section, decide: figure needed? (apply figure policy)
- [ ] Create Manim figure scripts for needed figures
- [ ] Render figures (dark + light themes)
- [ ] Rewrite sections following code-text integration pattern
- [ ] Verify code examples only use previously-introduced concepts
- [ ] Write summary table and bridge paragraph
- [ ] Translate to Chinese (`zh/tutorial/` counterpart)
- [ ] Verify both EN and ZH pages render with figures

## 9. Reference: Ch02 and Ch03 as Gold Standards

The definitive examples of these conventions in practice:

- **Ch02** (`tutorial/ch02-data-movement.md`): Data movement storyline.
  Recap of ch01 → concept intro (per-element vs block) → figure → tiled
  example → syntax-by-syntax explanation with figures → memory specifiers
  as supporting concept (not a separate chapter) → summary table → bridge.

- **Ch03** (`tutorial/ch03-parallelism.md`): Parallelism storyline.
  Recap of ch01+ch02 → concept intro (virtual parallelism) → figure →
  two-layer model → space specifiers as supporting detail (not a separate
  chapter) → shared reuse merged into specifiers section → type system
  aside → matmul application → tracing exercise → summary table → bridge.

When in doubt, re-read these two files and match their patterns.
