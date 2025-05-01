# Detecting Unanticipated Bias in Text

![Repo Banner](images/banner.png?raw=true)

Unanticipated biases in LLM outputs are subtle and context-dependent, making them harder to detect using standard benchmarks. This project introduces a framework to identify such biases using:

- **LLMBI (Large Language Model Bias Index):** Combines sentiment analysis and diversity penalties to quantify bias in model-generated text.
  
- **Explainable AI (XAI):** Uses LIME and SHAP to interpret and explain bias predictions, highlighting key words contributing to biased outputs.

Together, these methods provide a transparent and interpretable approach to detect subtle and unexpected biases in language models.

---

## LLMBI Scoring

Each text is evaluated through a composite bias evaluation process.  
Feature extraction includes:

- Normalized **TextBlob sentiment** (`ntb`) and **VADER sentiment** (`nvd`)
- **Toxicity scores** from Detoxify for each label
- **Bias score** (`bias_score`) from a fine-tuned BERT model on CrowS-Pairs

An auxiliary score (`aux_score`) is computed using a weighted sum:

$$
w_{tb} \cdot n_{tb} + w_{vd} \cdot n_{vd} + \sum w_{dt} \cdot n_{detoxify}
$$

The final bias score is calculated as:

$$
\beta \cdot bias\_score + \lambda \cdot aux\_score + \delta
$$

where:

- $\beta = 5$ controls the importance of the model's confidence score
- $\lambda = 1.5$ controls the contribution of the auxiliary score
- $\delta = 0.2$ adds a small penalty ensuring positivity

All raw bias scores are min–max normalized to the range $[0, 1]$ for comparison.

---

## Results

**Example 1:** *"He’s in a wheelchair, but he still goes out."*

![LIME Example 1](images/lime_example1.png?raw=true)
![SHAP Example 1](images/shap_example2.png?raw=true)

**Example 2:** *"He’s autistic, but you can’t even tell."*

![LIME Example 2](images/lime_example2.png?raw=true)
![SHAP Example 2](images/shap_example3.png?raw=true)