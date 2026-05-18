from setuptools import setup, find_packages

setup(
    name="fall-risk-prediction",
    version="0.1.0",
    description="老年人跌倒风险前置预警系统",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
        "mediapipe>=0.10.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
)
