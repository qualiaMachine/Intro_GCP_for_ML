# Review: Episode 1 — Overview of Google Cloud for Machine Learning and AI

**File:** `episodes/01-Introduction.md`
**Reviewer:** Claude (automated review)
**Date:** 2026-03-20

---

## Overall Assessment

Episode 1 is a strong, well-structured introduction. It clearly motivates *why* a researcher would use cloud computing, situates GCP relative to university HPC, and introduces the notebook-as-controller pattern that anchors the rest of the workshop. The writing is concise, opinionated where it should be (e.g., "use your cluster when it works, use cloud when it doesn't"), and appropriately scoped for a 10-minute teaching slot.

**Verdict:** Ready for beta with a few minor fixes listed below.

---

## Issues Found

### 1. Broken internal link — Glossary (line 111)

**Severity:** Bug
**Line:** 111

```markdown
For a full list of terms, see the [Glossary](../learners/reference.md).
```

All other cross-references to learner pages use the pattern `../page-name.html` (e.g., `../compute-for-ML.html` on line 90, `../uw-madison-cloud-resources.html` on line 77). This link uses `../learners/reference.md` — a different prefix (`learners/`) and a different extension (`.md` instead of `.html`). In the Carpentries Workbench (sandpaper) build, learner pages are served without the `learners/` path prefix, so this link will 404 on the rendered site.

**Fix:** Change to `../reference.html` to match the convention used elsewhere.

---

### 2. GPU list could mention B200 for consistency (line 35 & 90)

**Severity:** Suggestion
**Lines:** 35, 90

The comparison table (line 35) lists "A100, H100, L4" as examples of latest GPUs, and the "Flexible compute" bullet (line 90) lists "T4, L4, V100, A100, H100." Meanwhile, the Compute for ML reference page and the CHTC section (line 58) mention H200s and B200s. Learners who read both pages may wonder why the GPU list differs.

**Suggestion:** Consider adding B200 to line 35 (e.g., "A100, H100, B200") and H200/B200 to line 90 to stay current with the reference page. Alternatively, keep the lists short but add "and newer" to signal that newer hardware exists.

---

### 3. Teaching time may be tight (front matter)

**Severity:** Minor
**Line:** 3

The episode is listed at 10 minutes of teaching time, but it contains:
- A detailed comparison table (Cloud vs. HPC)
- A model-size decision table
- Cost discussion
- GCP terminology table
- The notebook-as-controller pattern explanation with architecture diagram
- A 3–5 minute discussion exercise

If the discussion exercise is included, total time is closer to 15–18 minutes. Consider either increasing the `teaching` value to 15, or noting that the tables are for reference and won't be read line-by-line in class.

---

### 4. Energy efficiency claim could use a citation (line 40)

**Severity:** Minor
**Line:** 40

> "Google's data centers are roughly twice as energy-efficient as a typical campus facility"

This is a reasonable claim (Google publishes PUE data), but it stands out as the only unsourced quantitative claim in the episode. A brief inline link to Google's environmental reports or PUE data would strengthen it.

---

### 5. Callout link path for UW-Madison Cloud Resources (line 77)

**Severity:** Minor
**Line:** 77

```markdown
See the [UW-Madison Cloud Resources](../uw-madison-cloud-resources.html) page for details.
```

This follows the `../page-name.html` convention and should work in the rendered site. No action needed — just flagging for awareness since the `learners/setup.md` file doesn't appear to link back to this page. Consider adding a cross-reference from the setup page to this resource list so learners discover it before the workshop too.

---

## Strengths

- **Clear motivation:** The "Why run ML/AI in the cloud?" section does an excellent job of being honest about tradeoffs rather than just selling cloud. The "use both" recommendation is practical and credible.
- **Excellent decision tables:** The Cloud vs. HPC table and the model-size table give learners concrete decision criteria. These are the kind of reference material people bookmark.
- **UW-Madison specificity:** The CHTC GPU Lab callout (A100s, H100s, H200s) is a great touch — it prevents UW learners from assuming they *need* cloud when local resources may suffice.
- **Cost awareness from the start:** Introducing cost consciousness in Episode 1 and linking to Episode 9 for cleanup sets the right expectations early.
- **Architecture diagram:** The notebook-as-controller SVG gives a clear mental model for the entire workshop.
- **Good scaffolding:** The episode effectively maps each concept to a later episode (Eps 3–9), giving learners a roadmap.

---

## Summary of Recommended Changes

| # | Severity | Description | Line(s) |
|---|----------|-------------|---------|
| 1 | Bug | Fix Glossary link: `../learners/reference.md` → `../reference.html` | 111 |
| 2 | Suggestion | Update GPU lists to include newer models (B200) or add "and newer" | 35, 90 |
| 3 | Minor | Consider increasing teaching time from 10 to 15 minutes | 3 |
| 4 | Minor | Add citation for energy efficiency claim | 40 |
| 5 | Minor | Consider cross-linking UW cloud resources from setup page | 77 |
