---
description: Refresh PROGRESS.md from git tags and the last gate run
---
1. Run: git tag --list 'phase-*' and git log --oneline -5
2. Rewrite PROGRESS.md as a table: phase | status (PASS or TODO) | date | note. A phase is PASS only if its tag phase-N-pass exists.
3. Print the table.
