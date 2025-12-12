from setuptools import setup, find_packages

setup(
    name="deep-learning-pipeline",
    version="1.0.0",
    description="Deep Learning Pipeline for Lecture 3",
    author="Deep Learning Course",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
        "gradio>=3.0.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "dl-pipeline=scripts.run_pipeline:main",
            "dl-train=scripts.train_model:main",
            "dl-download=scripts.download_data:main",
            "dl-ui=scripts.launch_ui:main",
        ],
    },
)