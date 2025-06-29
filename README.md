# Credit Scoring - Business Understanding

## 1. Basel II’s Requirement for Interpretability and Documentation

Basel II introduces a framework where banks using internal models must ensure transparency, documentation, and proper governance. Under the Internal Ratings-Based (IRB) approach, banks estimate risk components like Probability of Default (PD), Exposure at Default (EAD), and Loss Given Default (LGD). These components must be derived using models that are interpretable and verifiable.

Regulators (e.g., HKMA and World Bank) emphasize the importance of models that are explainable, stable over time, and documented to ensure that internal credit risk assessments are both trustworthy and compliant. These guidelines also apply to models using alternative or behavioral data.

Models that cannot be explained to risk committees or regulators are not suitable for regulated financial environments.

## 2. Necessity and Risks of Using a Proxy Default Label

When real default data is not available, it is common to use a proxy label to enable supervised learning. For example, a user becoming inactive, having negative balances, or account closure within a given period can serve as a proxy for default.

However, creating a proxy label introduces several risks:

- **Label noise**: Proxies may incorrectly label non-defaulting users as bad (false positives) or vice versa (false negatives).
- **Bias**: Proxy defaults may correlate with customer characteristics (e.g., geography or demographic) that lead to biased or unfair predictions.
- **Regulatory risk**: If the label does not reflect actual risk behavior, the model may fail to meet regulatory or internal audit standards.

To mitigate these risks, it is important to carefully define the proxy, test it for correlation with true business outcomes, and validate it across different customer segments.

## 3. Trade-off Between Logistic Regression + WoE and Gradient Boosting Models

**Logistic Regression + Weight of Evidence (WoE)**

- Interpretable: Coefficients and variable binning make it easy to understand.
- Documentable: Well-established in regulatory practice.
- Works well for linear relationships.
- Easy to implement bias detection and monotonicity constraints.
- Often used in Basel-compliant scorecards.

**Gradient Boosting (e.g., XGBoost, LightGBM)**

- Higher accuracy in many complex, nonlinear datasets.
- Difficult to interpret without SHAP or LIME.
- Requires more effort to document and monitor.
- Less transparent for internal audit and regulators.
- Needs post-hoc methods to explain predictions.
