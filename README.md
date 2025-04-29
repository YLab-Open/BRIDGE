## BRIDGE (Benchmarking Large Language Models for Understanding Real-world Clinical Practice Text)

## 📜 Background

Recent advances in **Large Language Models (LLMs)** have demonstrated transformative potential in **healthcare**,  yet concerns remain around their reliability and clinical validity across diverse clinical tasks, specialties, and languages. To support timely and trustworthy evaluation, building upon our [systematic review](https://ai.nejm.org/doi/full/10.1056/AIra2400012) of global clinical text resources, we introduce **[BRIDGE](https://arxiv.org/abs/2504.19467)**, **a multilingual benchmark that comprises 87 real-world clinical text tasks spanning nine languages and more than one million samples**. Furthermore, we construct this leaderboard of LLM in clinical text understanding by systematically evaluating **52 state-of-the-art LLMs** (by 2025/04/29).

More Details can be found in our [BRIDGE paper](https://arxiv.org/abs/2504.19467) and [systematic review](https://ai.nejm.org/doi/full/10.1056/AIra2400012).

## 🛠️ How to Use?

#### 1. Download the BRIDGE Dataset
All fully open-access datasets in BRIDGE are available in [BRIDGE-Open](https://huggingface.co/datasets/YLab-Open/BRIDGE-Open). To ensure the fairness of this leaderboard, we publicly release the following data for each task: Five completed samples serve as few-shot examples, and all testing samples with instructions and input information. Due to privacy and security considerations of clinical data, regulated-access datasets can not be directly published. However, all detailed task descriptions and their corresponding data sources are available in our [BRIDGE paper](https://arxiv.org/abs/2504.19467). Importantly, all 87 datasets have been verified to be either fully open-access or publicly accessible via reasonable request.

#### 2. LLM Inference

- Put the downloaded data into the `dataset_raw` folder.
- Edit the `BRIDGE.yaml` file to specify the tasks you want to evaluate.
- Edit the `run.sh` file to specify the model you want to evaluate.
- Run the `run.sh` file to start the evaluation, which will automatically load the model and run inference on the specified tasks (`main.py`).

#### 3. Evaluation

- Result folder: All inference results will be saved in the `result` folder, which will be automatically created. The structure of the result folder is as follows: result -> task -> model -> experiment
- Result extraction: We develop an automated script for each task separately to extract results from the standardized LLM outputs; details can be found in the `dataset` folder: `classification.py`, `extraction.py`, and `generation.py`.
- Evaluation metrics: We provide an evaluation function for different tasks; details can be found in the `metric` folder: `classification.py`, `extraction.py`, and `generation.py`.
- Evaluation script: run `evaluate_BRIDGE.py` to evaluate all tasks. The performance of each task will be saved in the `performance` folder. 

#### 4. Update the Leaderboard

If you would like to submit your model results to BRIDGE and demonstrate its performance, please send the genereated `result` folder to us, and we will update the leaderboard accordingly. The leaderboard will be updated regularly, and we will notify you via email once your results are added.



## 🤝 Contributing
We welcome and greatly value contributions and collaborations from the community!
If you have clinical text datasets that you would like to share for broader exploration, please contact us!
We are committed to expanding BRIDGE while strictly adhering to appropriate data use agreements and ethical guidelines. Let's work together to advance the responsible application of LLMs in medicine!


## 📢 Updates

- 🗓️ **2025/04/28**: BRIDGE Leaderboard V1.0.0 is now live!
- 🗓️ **2025/04/28**: Our paper [BRIDGE](https://arxiv.org/abs/2504.19467) is now available on arXiv!


## 📬 Contact Information

If you have any questions about BRIDGE or the leaderboard, feel free to reach out!

- **Leaderboard Managers**: Jiageng Wu (jiwu7@bwh.harvard.edu), Kevin Xie (kevinxie@mit.edu)
- **Benchmark Managers**: Jiageng Wu (jiwu7@bwh.harvard.edu), Bowen Gu (bogu@bwh.harvard.edu)
- **Program Lead**: Jie Yang (jyang66@bwh.harvard.edu)



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
