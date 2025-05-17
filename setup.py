import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Movie_Review_Sentiment_Analysis"
AUTHOR_USER_NAME = "Likithsatya192"
SRC_REPO = "sentiment-analysis"
AUTHOR_EMAIL = "likithsatya192@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for Movie Review Sentiment Analysis",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)