# Adaptive Online Intrusion Detection for Large-Scale IoT Networks Using Concept Drift Learning

## Overview
The rapid growth of Internet of Things (IoT) networks has introduced significant security challenges due to their large scale, dynamic behavior, and resource constraints. Traditional intrusion detection systems (IDS) are often static and fail to adapt to evolving attack patterns, leading to degraded performance over time.

This project presents an **adaptive online intrusion detection framework** designed for **large-scale IoT networks**, incorporating **concept drift learning** to dynamically adjust to changing network behaviors. The system continuously updates its models in real time, enabling robust and long-term intrusion detection in non-stationary environments.

---

## Key Features
- **Online Learning Framework** for real-time intrusion detection  
- **Concept Drift Awareness** to handle evolving attack patterns  
- **Adaptive Model Updating** without retraining from scratch  
- **Scalable Design** suitable for large-scale IoT environments  
- **Automated Performance Evaluation** across streaming data  

---

## Dataset
This project uses the **GothamDataset2025**, a large-scale simulated IoT network dataset designed for intrusion detection and cybersecurity research.

Dataset characteristics:
- Realistic large-scale IoT traffic simulation  
- Normal and multiple attack scenarios  
- Suitable for online and streaming-based learning  

---

## Methodology
1. **Data Preprocessing**  
   - Feature extraction and normalization  
   - Stream-based data handling  

2. **Online Intrusion Detection**  
   - Incremental model training on streaming data  
   - Continuous prediction and evaluation  

3. **Concept Drift Handling**  
   - Drift-aware learning strategies  
   - Adaptive updates to maintain detection accuracy  

4. **Evaluation Metrics**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  

---

## Technologies Used
- **Python 3**
- **River** (Online Machine Learning)
- **Scikit-learn**
- **NumPy**
- **Pandas**
- **Matplotlib**

---

## Project Structure
```text
adaptive-online-iot/
│
├── iotsim-air-quality-1.csv/(data)                  # Dataset and preprocessing scripts
├── models/                # Online learning and drift-aware models
├── results/               # Output results and plots
├── README.md
└── requirements.txt
