# Initial prompt

The common task instruction adapts the released AAAI 2026 baseline from [Liu et al.](https://ojs.aaai.org/index.php/AAAI/article/view/40496). Only speech input and output formatting are adapted. The original source reference and license are retained under `docs/references/`.

```text
The entire math problem is spoken in the audio.

Let's think step by step and output the final answer on a separate last answer line in the format FINAL: <answer>.
Please solve the problem strictly based on the information provided. Do not introduce any additional assumptions.
If you believe the problem lacks sufficient information or is unsolvable, explain why on a separate line in the format REASON: <your explanation here>, then output FINAL: I don't know.
Place REASON: before FINAL: and output exactly one FINAL: line.
```

Vanilla and Gate use identical initial task bytes. Native chat delimiters are supplied by each model wrapper.
