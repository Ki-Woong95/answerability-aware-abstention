# Third-party sources

## MOSS-Audio

Source: https://github.com/OpenMOSS/MOSS-Audio

Pinned upstream commit: `66326e6e0db34f036c86a76ba005efa4830c69dd`.

`MOSS-Audio/src` is vendored from the experimental checkout, including its existing local model integration changes. Publication formatting does not modify this vendored code. The upstream README credits the authors and states that model releases use Apache-2.0. The pinned source checkout does not contain the LICENSE file referenced by its pyproject; no missing upstream code license is invented here. Confirm upstream code licensing before a public redistribution.

## TreeCut

Source: https://github.com/j-bagel/treecut-math

Pinned upstream commit: `b48f3de5586d00fffaf3ae2392c6d16497b664ea`.

Only the generator Python modules are vendored under `vendor/treecut_math/`. They remain unmodified. Its Apache-2.0 license is retained in that directory. Dataset samples and the upstream Git history are excluded.

## AAAI abstention prompt

Reference: Liu et al., *Answering the Unanswerable Is to Err Knowingly*, AAAI 2026.

Released prompt source: https://github.com/nju-websoft/AbstentionReasoning at commit `2fab9640451dfd4b51ad3f342213bd926c1cffe1`.

The prompt is adapted for spoken math input and REASON/FINAL formatting. Source attribution, the released prompt reference and GPL-3.0 license text are retained under `docs/references/`. The authors' executable baseline script is not included.

Model weights and datasets have separate upstream terms and are not distributed here. No new project-wide license is assigned by this preparation.
