# Third-party notices

These notices cover the benchmark material in `cache/questions.csv`. They do
not assign a license to the authors' code or to the repository as a whole.

The cache contains selected examples converted to a shared CSV schema, with
local question IDs and train/test assignments. The source splits below refer
to the upstream datasets, not the local assignments.

| Dataset | Source split | Cached rows | Treatment |
| --- | --- | ---: | --- |
| MMLU | test | 240 | Six subjects, 40 questions each; choices and answers mapped to A-D. |
| GSM8K | main/train | 120 | Question retained; final numeric answer extracted from the worked solution. |
| ARC-Challenge | test | 600 | Four-choice items selected; choice labels and answers mapped to A-D. |
| HellaSwag | validation | 600 | Question and choice text excluded; IDs, labels, split metadata and numeric experiment caches retained. |

## MMLU

Dataset: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu).
Credit: Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika,
Dawn Song and Jacob Steinhardt, *Measuring Massive Multitask Language
Understanding* (2021).

The selected subjects are world religions, professional law, high school
biology, prehistory, marketing and high school geography.

[Upstream MIT license](https://github.com/hendrycks/test/blob/master/LICENSE):

```text
MIT License

Copyright (c) 2020 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GSM8K

Dataset: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k).
Credit: Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun,
Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano,
Christopher Hesse and John Schulman, *Training Verifiers to Solve Math Word
Problems* (2021).

[Upstream MIT license](https://github.com/openai/grade-school-math/blob/master/LICENSE):

```text
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ARC-Challenge

Dataset: [Allen Institute for AI, AI2 Reasoning Challenge
(ARC)](https://huggingface.co/datasets/allenai/ai2_arc).
Credit: Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal,
Carissa Schoenick and Oyvind Tafjord, *Think you have Solved Question Answering?
Try ARC, the AI2 Reasoning Challenge* (2018).

The ARC portions of `cache/questions.csv`, including their schema adaptations,
are distributed under [Creative Commons Attribution-ShareAlike 4.0
International](https://creativecommons.org/licenses/by-sa/4.0/)
([legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode)), as
specified by AllenAI's dataset card. Changes consist of selecting 600
four-choice test items, mapping their options and answer labels to A-D, and
adding local question IDs and split assignments. The question and choice text
is retained. Attribution, license and change notices must accompany further
sharing; adaptations of this ARC material must use the same or a compatible
license. No endorsement by AllenAI or the dataset authors is implied.

## HellaSwag

Experiment source: [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag).
Credit: Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi and Yejin Choi,
*HellaSwag: Can a Machine Really Finish Your Sentence?* (2019).

The public cache leaves `question`, `A`, `B`, `C` and `D` empty for all 600
HellaSwag rows. It retains their IDs, answer labels, subject and split metadata,
and the numeric experiment caches used to reproduce the reported analyses.
Text-dependent retraining or embedding regeneration requires a separately
supplied dataset that the user is entitled to use.

The Hugging Face dataset card identifies an MIT license and links to the
original repository. GitHub has since disabled that repository following a
[wikiHow copyright takedown notice dated September 14,
2026](https://github.com/github/dmca/blob/master/2026/09/2026-09-14-wikihow.md).
The source is cited here for experimental provenance; this release does not
redistribute HellaSwag question or choice text.
