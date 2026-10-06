---
description: Check the implementation of a paper section against the paper (e.g. /check-against-paper 3.3)
argument-hint: <section, e.g. 3.2 | 3.3 | 3.4 | 4 | Alg1 | Eq9>
---

Check the code that implements paper section **$ARGUMENTS** of CacheScout.

1. Read the relevant part of `docs/paper_notes.md` (and the PDF text for section $ARGUMENTS if the notes are thin).
2. Find the code implementing it (grep for the section/equation number in docstrings, then related names).
3. Use the `paper-checker` agent on those files, scoped to section $ARGUMENTS.
4. Report: what matches, what deviates (quote the paper), invented details, and any ambiguity that should be added
   to `docs/open_questions.md`. Do not modify code unless I ask.
