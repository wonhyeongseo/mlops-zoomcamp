# A Journey through MLflow: Experiment Tracking and Model Management

This repository contains the code and resources for the technical blog post titled [A Journey through MLflow: Experiment Tracking and Model Management](https://won.hashnode.dev/mlops-zoomcamp-week-2) . The blog post explores the powerful capabilities of MLflow, a tool for experiment tracking and model management. It also includes an addendum that introduces DAGsHub and demonstrates how to use it with Google Colab for enhanced collaboration and version control.

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make dirs` or `make clean`
    ├── README.md          <- The top-level README for developers using this project.
    ├── homework.ipynb     <- Code submission
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw.dvc        <- DVC file that tracks the raw data
    │   └── raw            <- The original, immutable data dump
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── preprocess_data.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── register_model.py
    │   │   ├── train.py
    │   │   └── hpo.py
    │   │
    │   └── img           <- Attached images
    │       ├── best-model.jpg
    │       ├── contour-plot.jpg
    │       └── hpo.jpg
    │
    └── dvc.yaml           <- Defining the data pipeline stages, dependencies, and outputs.


--------

## Instructions

To reproduce the experiments and follow along with the blog post, please follow these steps:

1. Open the `[notebooks/homework.ipynb](https://dagshub.com/wonhyeongseo/mlops-week2/src/main/notebooks/homework.ipynb)` Jupyter Notebook by clicking the `Open with Google Colab` button.
3. Execute the code cells in the notebooks to run the experiments, track the results, and explore the functionalities of MLflow and DAGsHub.
4. Refer to the blog post for detailed explanations and insights regarding each step of the process.

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/home/): Official documentation for MLflow.
- [DAGsHub Documentation](https://docs.dagshub.com/): Official documentation for DAGsHub.
- [Google Colab](https://colab.research.google.com/): Cloud-based Jupyter notebook environment provided by Google.

## Conclusion

This repository provides a comprehensive guide to implementing experiment tracking and model management using MLflow. It also introduces DAGsHub for enhanced collaboration and version control in data science projects. By following the instructions and exploring the provided resources, you can gain a deeper understanding of these powerful tools and apply them to your own projects.

For more details and insights, please refer to the accompanying blog post, which offers a step-by-step walkthrough of the experiments and highlights the key features of MLflow and DAGsHub.

Happy tracking and modeling!

---
*Note: This repository is part of a homework assignment and is intended for educational purposes.*