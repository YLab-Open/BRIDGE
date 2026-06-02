## BRIDGE (Benchmarking Large Language Models for Understanding Real-world Clinical Practice Text)

## 📜 Background

**Large Language Models (LLMs)** have demonstrated transformative potential in **healthcare**,  yet concerns remain around their reliability and clinical validity across diverse clinical tasks, specialties, and languages. To support timely and trustworthy evaluation, building upon our [systematic review](https://ai.nejm.org/doi/full/10.1056/AIra2400012) of global clinical text resources, we introduce **[BRIDGE](https://arxiv.org/abs/2504.19467)**, **a multilingual benchmark that comprises 87 real-world clinical text tasks spanning nine languages and more than one million samples**. The first leaderboard release systematically evaluated **52 state-of-the-art LLMs**; the leaderboard has since been expanded through regular updates.

**Key Features**: Real-world Clinical Text, 9 Languages, 9 Task types, 14 Clinical specialties, 7 Clinical document types, 20 Clinical applications covering 6 clinical stages of patient care.

**More Details** can be found in our [BRIDGE paper](https://arxiv.org/abs/2504.19467) and [systematic review](https://ai.nejm.org/doi/full/10.1056/AIra2400012), and the comprehensive leaderboard is available at [BRIDGE Leaderboard](https://huggingface.co/spaces/YLab-Open/BRIDGE-Medical-Leaderboard).

**This project** is led and maintained by the team of <a href="https://ylab.top/">Prof. Jie Yang</a> and <a href="https://www.drugepi.org/team/joshua-kueiyu-lin">Prof. Kueiyu Joshua Lin</a> at Harvard Medical School and Brigham and Women's Hospital.

## 📢 Updates

- 🗓️ **2026/04/08**: We continue to evaluate the latest models. BRIDGE Leaderboard now includes **107** evaluated models.
- 🗓️ **2025/06/03**: Updated the leaderboard with 21 additional models, for 75 evaluated models in total.
- 🗓️ **2025/04/28**: BRIDGE Leaderboard V1.0.0 is now live with 54 evaluated models.
- 🗓️ **2025/04/28**: Our paper [BRIDGE](https://arxiv.org/abs/2504.19467) is available on arXiv.

## 🛠️ How to Use?

### 1. Download the BRIDGE Dataset

All fully open-access datasets in BRIDGE are available in [BRIDGE-Open](https://huggingface.co/datasets/YLab-Open/BRIDGE-Open). To ensure leaderboard fairness, we publicly release five completed few-shot examples for each task and all testing samples with instruction/input fields. Regulated-access clinical datasets cannot be directly published, but task descriptions and original data sources are listed in the [BRIDGE paper](https://arxiv.org/abs/2504.19467).

Put task files under `dataset_raw/`:

```text
dataset_raw/<task_name>.SFT.json
dataset_raw/example/<task_name>.example.json
dataset_raw/example_cot/<task_name>.example.json
```

### 2. Prepare Local Configs

Real API keys and local checkpoint paths are intentionally not committed. Create
local files from the templates:

```bash
cp configs/API.key.example.yaml configs/API.key.yaml
cp configs/dict_model_path.example.json configs/dict_model_path.json
```

`configs/dict_model_path.json` maps model names to local checkpoint paths. `configs/API.key.yaml`
stores provider credentials, including the `AZURE-OPENAI` block used by the Azure
example notebook.

### 3. Run Local Inference

The recommended local runtime is vLLM:

```bash
mamba activate vllm
python main.py \
  --model_name gpt-oss-20b \
  --gpus 0,1 \
  --engine vllm \
  --config configs/BRIDGE.yaml \
  --model_file configs/dict_model_path.json
```

Configure tasks, data paths, result paths, prompt modes, decoding modes, and
token budgets in `configs/BRIDGE.yaml`. Supported prompt modes are `direct`, `cot`,
`direct-N-shot`, and `cot-N-shot`.

Results are saved as:

```text
result/<task_name>/<model_name>/<task_name>-<prompt_mode>-<decoding>-<seed>.result.json
```

### 4. Evaluate Results

Evaluation is configured separately from inference through
`evaluation/evaluate_BRIDGE.yaml`:

```bash
python -m evaluation.bridge
python -m evaluation.bridge --config evaluation/evaluate_BRIDGE.yaml
```

Set models, prompt modes, task types, and task subsets in the YAML file. The
evaluation code warns when a result file does not align with the expected task
rows, which supports partial sampling runs while keeping data-integrity issues
visible.

### 5. More Documentation

- `USAGE.md`: detailed runtime, data, inference, and evaluation guide.
- `configs/README.md`: runtime config templates and local-only config paths.
- `scripts/README.md`: single-model, multi-model, and delayed launch scripts.
- `notebooks/README.md`: interactive notebook examples.
- `dataset/README.md`: task data loading and prompt setup.
- `model/README.md`: local inference and provider batch helpers.
- `evaluation/README.md`: maintained evaluation entry point and YAML examples.
- `tests/README.md`: lightweight regression tests.

### 6. Update the Leaderboard

To submit model results to [BRIDGE Leaderboard](https://huggingface.co/spaces/YLab-Open/BRIDGE-Medical-Leaderboard), send the generated `result` folder to the maintainers. We will update the leaderboard regularly and notify you when results are added.

## 🤝 Contributing
We welcome and greatly value contributions and collaborations from the community!
If you have clinical text datasets that you would like to share for broader exploration, please contact us!
We are committed to expanding BRIDGE while strictly adhering to appropriate data use agreements and ethical guidelines. Let's work together to advance the responsible application of LLMs in medicine!

## 🚀 Donation

BRIDGE is a non-profit, researcher-led benchmark that requires substantial resources (e.g., high-performance GPUs, a dedicated team) to sustain. To support open and impactful academic research that advances clinical care, we welcome your contributions. Please contact Prof. Jie Yang at <jyang66@bwh.harvard.edu> to discuss donation opportunities.


## 📬 Contact Information

If you have any questions about BRIDGE or the leaderboard, feel free to reach out!

- **Leaderboard Managers**: Jiageng Wu (<jiwu7@bwh.harvard.edu>), Kevin Xie (<kevinxie@mit.edu>), Bowen Gu (<bogu@bwh.harvard.edu>)
- **Benchmark Managers**: Jiageng Wu, Bowen Gu
- **Project Lead**: Jie Yang (<jyang66@bwh.harvard.edu>)



## 📚 Citation
If you find this leaderboard useful for your research and applications, please cite the following papers:

```bibtex
@article{BRIDGE-benchmark,
    title={BRIDGE: Benchmarking Large Language Models for Understanding Real-world Clinical Practice Text},
    author={Wu, Jiageng and Gu, Bowen and Zhou, Ren and Xie, Kevin and Snyder, Doug and Jiang, Yixing and Carducci, Valentina and Wyss, Richard and Desai, Rishi J and Alsentzer, Emily and Celi, Leo Anthony and Rodman, Adam and Schneeweiss, Sebastian and Chen, Jonathan H. and Romero-Brufau, Santiago and Lin, Kueiyu Joshua and Yang, Jie},
    year={2025},
    journal={arXiv preprint arXiv: 2504.19467},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2504.19467},
}
@article{clinical-text-review,
    title={Clinical text datasets for medical artificial intelligence and large language models—a systematic review},
    author={Wu, Jiageng and Liu, Xiaocong and Li, Minghui and Li, Wanxin and Su, Zichang and Lin, Shixu and Garay, Lucas and Zhang, Zhiyun and Zhang, Yujie and Zeng, Qingcheng and Shen, Jie and Yuan, Changzheng and Yang, Jie},
    journal={NEJM AI},
    volume={1},
    number={6},
    pages={AIra2400012},
    year={2024},
    publisher={Massachusetts Medical Society}
}
```
If you use the datasets in BRIDGE, please also cite the original paper of datasets, which can be found in our BRIDGE paper.

<div style="display: flex; align-items: center; justify-content: space-between; width: 100%; height: 150px;">
  <img
    src="https://cdn-uploads.huggingface.co/production/uploads/67a040fb6934f9aa1c866f99/1bNk6xHD90mlVaUOJ3kT6.png"
    alt="HMS"
    style="width: 20%; height: 100%; object-fit: contain;"
  />
  <img
    src="https://cdn-uploads.huggingface.co/production/uploads/67a040fb6934f9aa1c866f99/ZVx7ahuV1mVuIeygYwirc.png"
    alt="MGB"
    style="width: 36%; height: 100%; object-fit: contain;"
  />
  <img
    src="https://cdn-uploads.huggingface.co/production/uploads/67a040fb6934f9aa1c866f99/TkKKjmq98Wv_p5shxJTMY.png"
    alt="Broad"
    style="width: 19%; height: 100%; object-fit: contain;"
  />
  <img
    src="https://cdn-uploads.huggingface.co/production/uploads/67a040fb6934f9aa1c866f99/UcM8kmTaVkAM1qf3v09K8.png"
    alt="YLab"
    style="width: 15%; height: 100%; object-fit: contain;"
  />

</div>
