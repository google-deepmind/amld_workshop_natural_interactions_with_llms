# AMLD Workshop - Natural Interactions with Foundation Models
[![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/amld_workshop_natural_interactions_with_llms/blob/master/AMLD_Ink_Interactions.ipynb)

This repository contains the instructions and the code related to the
[AMLD 2025 Workshop](https://2025.appliedmldays.org/) on natural interactions
with foundation models. The goal of the practical session is to highlight how
digital ink can be used as a novel manner to interact with visual large language
models. More specifically, participants will first interact with a fine-tuned
version of [PaliGemma 2](https://arxiv.org/pdf/2412.03555) and see in
application how ink can be used for natural interactions. In the second part,
participants will try prompting the model to understand the limitations of
adding a new out-of-distribution gesture. Finally, participants will generate
synthetically a dataset for a new gesture and fine-tune the model to support
it - the resulting model should have learned a new capability without forgetting
the previous ones.

To load, interact, and fine-tune PaliGemma 2, we will be using the
[big_vision](https://github.com/google-research/big_vision) framework. The
fine-tuned PaliGemma 2 has used ink data from
[DeepWriting](https://github.com/emreaksan/deepwriting) and
[IAM-OnDB](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)\[1\],
and standard Wikipedia pages as documents.

\[1\] U. Marti and H. Bunke. The IAM-database: An English Sentence Database for
Off-line Handwriting Recognition. Int. Journal on Document Analysis and
Recognition, Volume 5, pages 39 - 46, 2002.

## Installation

Open up the following notebook [![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/amld_workshop_natural_interactions_with_llms/blob/master/AMLD_Ink_Interactions.ipynb) - everything is there!

## Contributors

This workshop would not have been possible without the DigInk support! The
contributors are listed in alphabetical order:

-   Anastasiia Fadeeva
-   Blagoj Mitrevski
-   Claudiu Musat
-   Diego Antognini
-   Leandro Kieliger
-   Philippe Schlattner
-   Vincent Coriou

## Citing this work

Please consider citing this work if you find the repository helpful.

```
@misc{sky_t1_2025,
  author       = {DigInk Team},
  title        = {AMLD Workshop - Natural Interactions with Foundation Models},
  howpublished = {https://github.com/google-deepmind/amld_workshop_natural_interactions_with_llms},
  note         = {Accessed: 2025-02-14},
  year         = {2025}
}
```

## License and disclaimer

Copyright 2025 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
