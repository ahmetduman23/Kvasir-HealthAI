from setuptools import setup, find_packages

def read_requirements(path: str = "requirements.txt"):
    try:
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

setup(
    name="kvasir-healthai",
    version="0.1.0",
    author="Ahmet Yasir Duman",
    author_email="ahmetyasirduman@gmail.com",
    description="Advanced U-Net pipeline for gastrointestinal polyp segmentation (Kvasir-SEG) with Explainable AI (Grad-CAM).",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ahmetduman23/kvasir-healthai",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    keywords=["unet", "segmentation", "medical imaging", "kvasir", "xai", "healthcare ai"],
)
