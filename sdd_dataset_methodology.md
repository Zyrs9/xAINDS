# 4. Dataset Engineering and Evaluation Methodology

The reliability of a Network Intrusion Detection System (NIDS) depends fundamentally on the quality, realism, and preparation of its data. For this Explainable AI-Driven Lightweight NIDS, the data pipeline is designed to simulate realistic Small and Medium-sized Business (SMB) network environments. Drawing from industry-standard datasets—CICIDS2017, CICIoT2023, and UNSW-NB15—the methodology prioritizes unsupervised anomaly detection using Isolation Forest and SHAP for explainability, strictly avoiding computationally heavy and unrealistic deep learning preprocessing techniques.

### 4.1. Dataset Preprocessing Pipeline

The initial data ingestion phase focuses on data integrity, minimizing noise without destroying the underlying network behavior characteristics. A lightweight, robust preprocessing workflow is employed to prepare the raw flow data:

*   **Dataset Cleaning & Deduplication:** Redundant traffic flows are removed to prevent the model from artificially skewing its decision boundaries toward repetitive network events. 
*   **Corrupted Data Handling:** Rows containing infinite values (e.g., resulting from division-by-zero in flow rate calculations) or malformed packets are strictly dropped to ensure numerical stability during model training.
*   **Missing Value Resolution:** Given the lightweight deployment constraints, complex generative imputation models are avoided. Features with excessive missing values are dropped, while sparse missing values in critical features are handled via simple median imputation to resist the influence of extreme outliers.

### 4.2. Handling Realistic Class Imbalance

In real-world SMB networks, malicious traffic constitutes a minute fraction of total network activity; the vast majority of traffic is entirely benign. Consequently, this project intentionally avoids perfect class balancing (e.g., exactly 50% benign and 50% malicious traffic). 

Artificially inflating the minority class (anomalies) through aggressive oversampling techniques (such as SMOTE or GANs) inherently distorts the natural statistical distribution of network traffic. Training on a perfectly balanced dataset often yields an unrealistic model that is hypersensitive and prone to unacceptably high False Positive Rates (FPR) in production. Instead, this methodology embraces a **controlled, realistic imbalance**. Because the core detection engine relies on an unsupervised Isolation Forest, the model is engineered to profile normal network behavior. Therefore, the training data remains heavily benign, accurately reflecting a standard network baseline. Conversely, the evaluation data maintains a realistic proportion of anomalies to rigorously test the model's discriminative anomaly-detection power under practical conditions.

### 4.3. Train-Test Splitting Methodology

To accurately validate the Isolation Forest's capabilities, the dataset splitting strategy must reflect the unsupervised nature of the deployment environment. The dataset is partitioned into distinct training and evaluation sets, adhering to stringent data isolation protocols to prevent overlap.

*   **Training Data Composition:** The training split consists almost exclusively of benign traffic. This allows the Isolation Forest to effectively map the boundaries of normal network operations without its isolation trees being skewed by known attack signatures.
*   **Evaluation Data Composition:** The test set serves as a realistic deployment simulation. It contains a mix of unseen benign traffic and the full spectrum of anomalies (attacks) present in the source datasets.
*   **Stratified Splitting:** Within the evaluation set, stratified sampling is employed to ensure that all attack vectors from the CICIDS2017, CICIoT2023, and UNSW-NB15 datasets are proportionally represented. This prevents bias toward high-volume brute-force or DDoS attacks over low-volume targeted exploits.
*   **Separation of Flows:** Traffic flows are partitioned cleanly to ensure the model does not memorize overlapping session data, thereby guaranteeing that the evaluation flows are entirely unseen by the training engine.

### 4.4. Feature Scaling and Data Leakage Prevention

Feature scaling is a mathematically critical step, as network features operate on drastically different magnitudes (e.g., total bytes transferred in the millions versus TCP flag counts in the single digits). Without scaling, features with larger numerical ranges would disproportionately dominate the Isolation Forest's path length calculations and distort SHAP explainability values.

For this architecture, **StandardScaler** is utilized. By centering the data around a mean of zero with a standard deviation of one, StandardScaler effectively normalizes feature magnitudes while preserving the relative distance of extreme outliers. This property is highly advantageous for an anomaly detection algorithm like Isolation Forest, which isolates observations based on their distance from the norm.

**Strict Prevention of Data Leakage:**
A pervasive academic and engineering error in machine learning is applying feature scaling across the entire dataset prior to splitting. This leaks statistical information (mean and variance) from the test set into the training phase, artificially inflating evaluation metrics and ruining real-world validity. To absolutely prevent data leakage, this methodology enforces a rigorous scaling protocol:

1.  **Fit on Training Only:** The StandardScaler is strictly `fitted` (calculating mean and standard deviation) **only** on the training dataset. The evaluation data is entirely excluded from this computation.
2.  **Transform Training Data:** The fitted scaler is then used to `transform` the training dataset into scaled values.
3.  **Reuse for Evaluation:** The exact same fitted scaler is subsequently reused to `transform` the test dataset. 

This protocol ensures that the test data is normalized strictly relative to the statistical baseline of the training data, authentically simulating a real-time production environment where future network traffic statistics are fundamentally unknown to the deployed NIDS.
