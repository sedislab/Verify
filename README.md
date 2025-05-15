# The VERIFY Dataset

VERIFY is a large-scale dataset for enabling formal language translation. It contains formal LTL formulas with their LaTeX representations, ITL (Intermediate Technical Language) representations, and natural language translations across different domains.

We used a formula enumerator to generate several LTL formulas, SPOT to verify the canonical forms of these formulas, a semantically valid rule-based approach to generate an intermediary form we call ITL, and [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) to to generate solutions. 
We also manually verified a randomly sampled 10,000 examples and used [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) to serve as a judge for approximately 18% of the samples.

## Dataset Statistics

- Total entries: 217916
- Unique formulas: 217423
- Domains:
  - Aerospace (16822 entries),
  - Automotive/Autonomous Vehicles (16712 entries),
  - Build Pipelines and CI/CD (16743 entries),
  - Financial/Transaction Systems (16769 entries),
  - Home Automation (16753 entries),
  - Industrial Automation/Manufacturing (16787 entries),
  - Medical Devices (16714 entries),
  - Networking/Distributed Systems (16751 entries),
  - Robotics (16801 entries),
  - Security and Authentication (16753 entries),
  - Smart Grid/Energy Management (16719 entries),
  - Version Control and Code Reviews (16823 entries),
  - Web Services/APIs (16769 entries)

## Dataset fields
The VERIFY dataset contains the following fields:

- **formula_id**: Unique identifier for each formula
- **formula**: The formula text
- **latex**: LaTeX representation of the formula
- **formula_canonical_form**: Canonical form of the formula
- **depth**: Depth of the formula
- **itl_id**: ID of the ITL representation
- **itl_text**: ITL representation text
- **generation_method**: Method used to generate the ITL representation
- **verified**: Whether the ITL representation has been verified
- **is_correct**: Whether the ITL representation is correct
- **translation_id**: ID of the translation
- **domain**: Domain of the natural language translation
- **activity**: Activity associated with the translation
- **translation**: Natural language translation of the formula
- **generation_time**: Time taken to generate the translation

## Improving NL-LTL Translation

To demonstrate the quality of this dataset, we train a series of T5 models trained on this data.


The models achieve great results on contextual, formal logic translation. We present Exact Match as a metric for NL to LTL translation and an Expert-Scored Likert Scale for LTL to NL translation. 
Please see our paper for more details on the evaluation setup.


## Reproducing our results

The pipeline we used to produce the data is fully open-sourced!

- [Code](https://github.com/sedislab/{verify-dataset-creation})
- [Dataset](https://huggingface.co/datasets/sedislab/VERIFY)


## Citation

If you find our work useful, please consider citing us!

```bibtex
@article{tag,
  title   = {VERIFY: The First Multi-Domain Dataset Grounding LTL in Contextual Natural Language via Provable Intermediate Logic},
  author  = {Quansah, Paapa Kwesi and Bonnah, Ernest and Rivas, Pablo Perea},
  year    = {2025},
  journal = { }
}
```

## Dataset Owner(s):
Secured and Dependable Intelligent Systems (SeDIS) Lab 

## Release Date:
05/09/2025

## Data Version
1.0 (05/09/2025)

## License/Terms of Use:
cc-by-4.0

## Intended Usage:
This dataset is intended to be used by the community to continue to improve models. The data may be freely used to train and evaluate.
