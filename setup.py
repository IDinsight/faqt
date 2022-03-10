from setuptools import setup, find_packages

requirements_path = "requirements.txt"

with open(requirements_path) as fp:
    requirements = fp.read().splitlines()

if __name__ == "__main__":
    setup(
        name="faqt",
        version="0.0.1",
        description="Add an NLP layer over your FAQs",
        url="#",
        author="idinsight",
        install_requires=requirements,
        author_email="",
        packages=find_packages(),
        include_package_data=True,
        python_requires=">=3.8",
        zip_safe=False,
    )
