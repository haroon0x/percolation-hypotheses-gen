# percolation-hypotheses-gen
This project aims to develop a hypothesis generation engine that explores the relationship between hypothesis complexity and information density, specifically focusing on identifying the "percolation point"—the theoretical limit of human comprehension where information density sharply declines despite increasing complexity.

## Overview
- There exists a correlation between the complexity of a hypothesis and its information density, up to a certain point. This point, the percolation point, represents the limit of human comprehension. Beyond this point, while complex hypotheses can still be generated (by LLMs), their information density plummets, leading to potentially misleading or nonsensical outputs (hallucination) that mimic scientific plausibility but lack grounding in reality. This project aims to visually and computationally demonstrate this concept.



## Hypothesis Generation
## Information density



```
percolation-hypotheses-gen
├─ .python-version
├─ app
│  └─ app.py
├─ Frontend
│  ├─ index.html
│  ├─ main.js
│  └─ style.css
├─ LICENSE
├─ log
├─ pyproject.toml
├─ README.md
├─ sample_pdfs
│  ├─ 2505.09053v1.pdf
│  ├─ 2505.09151v1.pdf
│  └─ 2505.11309v1.pdf
└─ src
   ├─ agents
   │  ├─ agent.py
   │  ├─ sub_agents.py
   │  └─ __init__.py
   ├─ config
   │  ├─ config.py
   │  └─ __init__.py
   ├─ Hypothesis_Analysis
   │  ├─ complexity_score.py
   │  ├─ entity_labels.py
   │  ├─ information_density.py
   │  └─ information_density_.py
   ├─ literature_processing
   │  ├─ process_literature.py
   │  └─ __init__.py
   ├─ main.py
   ├─ models
   │  ├─ model.py
   │  └─ __init__.py
   ├─ server.py
   ├─ utils
   │  ├─ data_utils.py
   │  ├─ hypothesis_utils.py
   │  ├─ visualization_utils.py
   │  └─ __init__.py
   └─ __init__.py

```