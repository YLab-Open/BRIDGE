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


<h2>📜 Background</h2>
<p>Recent advances in <strong>Large Language Models (LLMs)</strong> have demonstrated transformative potential in improving healthcare delivery and clinical research. By combining extensive pretraining with supervised instruction tuning across diverse tasks, LLMs excel in natural language understanding, generation, and reasoning. These capabilities allow LLMs to serve as versatile, general-purpose medical assistants.</p>
<p>Despite this promise, concerns remain around the <strong>reliability and clinical validity</strong> of LLM-generated outputs. Real-world contexts often involve unstructured, multilingual text from <strong>electronic health records (EHRs)</strong>, and require support for tasks like phenotype identification and event extraction that remain underexplored. Moreover, the scarcity of <strong>multilingual benchmarks</strong> further limits the global applicability of LLMs in medicine.</p>
<p>To address these challenges, we introduce the <em>largest multilingual clinical benchmark</em> to date, <strong>BRIDGE (Benchmarking Large Language Models in Multilingual Real-world Clinical Text Understanding)</strong>, evaluating 52 LLMs on:</p>
<ul>
  <li><strong>87 clinical tasks</strong></li>
  <li><strong>9 languages</strong></li>
  <li><strong>1M+ clinical samples</strong></li>
</ul>

<div style="text-align: center;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/67a040fb6934f9aa1c866f99/2fh-jETNSL9iXJXTT-fdN.png" style="width: 50%;" alt="BRIDGE benchmark graphic">
</div>

<h2>🌍 Key Features</h2>
<p>Our benchmark spans a wide range of document types and clinical tasks, including classification, event extraction, and generation. It further supports three inference strategies: <strong>zero-shot</strong>, <strong>few-shot</strong>, and <strong>chain-of-thought (CoT)</strong> prompting. We evaluated <strong>52 LLMs</strong>, including general-purpose, open-source, proprietary, and medical-domain models.</p>
<ul>
  <li><strong>Multilingual Data</strong>: Clinical tasks in <strong>9 languages</strong> for global relevance.</li>
  <li><strong>Diverse Clinical Documents</strong>: Notes, summaries, radiology reports, and more.</li>
  <li><strong>Multiple NLP Tasks</strong>: Classification, extraction, QA, summarization, etc.</li>
  <li><strong>Evaluation Modes</strong>:
    <ul>
      <li><strong>Zero-shot</strong></li>
      <li><strong>Few-shot</strong></li>
      <li><strong>Chain-of-Thought (CoT)</strong> reasoning</li>
    </ul>
  </li>
</ul>


<h2>🏆 BRIDGE Leaderboard</h2>
<p>To support ongoing evaluation, we introduce our <strong>BRIDGE Leaderboard</strong>, which provides:</p>
<ul>
  <li>Easy visualizations</li>
  <li>Side-by-side comparisons</li>
  <li>Continuous tracking of LLM performance across tasks, languages, and evaluation strategies</li>
</ul>
<p>This leaderboard empowers researchers and clinicians to make informed decisions and track model progress over time.</p>

<h2>📚 Citation</h2>
<pre><code>@article{BRIDGE2025,
  title     = {PAPER TITLE},
  author    = {Your Name and Contributors},
  year      = {2025},
  journal   = {Your Journal or Conference},
}
</code></pre>
