# iris-ml-cicd

A mini-project that demonstrates how to automate training and evaluation of a machine learning model using GitHub Actions CI/CD.

### Features

- 📦 Train on 30-row initial dataset.
- 🧠 Automatically retrain and evaluate when new data is pushed.
- 🔁 CI/CD fully managed with GitHub Actions.

### How it works

1. Push initial training data: triggers training only.
2. Add new data: triggers full retrain + evaluation.
