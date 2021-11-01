---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Learn to contribute documentation to napari.org

Following this tutorial you will learn the different types of napari.org documents, pick a type of document you'd like to submit and open a Pull Request to contribute your document.

## What you'll learn
- The different types of documentation on napari.org and how to decide which one to write
- The benefits of our documentation templates and where to find them
- How to open a Pull Request for your document and where to put it


## What you'll need
- An idea of something you'd like to teach users about
- A [GitHub account](https://github.com/) you can use to submit your Pull Request
- [Jupyter notebook](https://jupyter.org/index.html) and [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html)


## Document types
napari's documentation is based on the [Diátaxis Framework](https://diataxis.fr/) and consists of four main types of documents, each with their own section on the website.

![Diátaxis framework placing tutorials, how-to guides, explanations and references on a two axis diagram](https://diataxis.fr/_images/diataxis.png)

**Tutorials** are a contained lesson designed to help the user achieve understanding of a concept, feature, workflow or process through practice. They should be composed of a set of concrete, sequential steps that guide the user towards a clearly defined result. Typically, tutorials are sufficiently detailed to provide a basis for further learning without getting too tangled in explanations, and are aimed at users who don't necessarily know what they want to do and what's available to them.

**How-tos** are also concrete steps that help the user to achieve a defined result. Unlike tutorials, users following a how-to already know *what* they want to achieve, they're just not sure how to do it. How-tos should list steps with only the minimal detail required to repeat the steps. 

**Explanations** steer away from concrete practice-based lessons and focus on theoretical and concept understanding. They are inevitably more in depth than both tutorials and how-tos, and open up the floor for discussion. 

### Picking your document

New documentation might be useful when we develop a new viewer or layer interaction, add an entirely new feature to napari's API, change the way things are built under the hood or use napari for a novel application.

Where possible, small new features or interactions with layers or the viewer itself should be added to the existing tutorials and how-tos to make sure they are discoverable. For example, if a new layer button was added e.g. for Surface layers, this should be documented in the Viewer tutorial. If a new feature is fairly complex it might warrant its own document, e.g. hooking up your own callbacks to layer events or performance monitoring your Python scripts.

To pick your document, you should consider what information you want to communicate and to whom. For example "I want to document how this feature has been built to future developers, so that it's easy to maintain" - this sounds like you're wanting to communicate complex, in-depth information for developers, and would require an explanation style document. "I want to show all users how to install plugins" is a set of very quick steps that might warrant a how-to while "I want to give biologists an overview of how to perform segmentation in napari" is a broader lesson for which a tutorial would be suitable. Of course, you might want to write more than one document depending on who you're targetting.


## Documentation templates

Our goal is that all tutorials and how-tos are easily downloadable and executable by our users. This helps ensure that they are reproducible for our users and are more easily maintained. [Jupyter notebooks]() are a great option for our documents, because they allow you to easily combine code and well formatted text in markdown. However, their [raw JSON format]() is not great for version control, so we use [MyST Markdown]() documents in our repository and on napari.org.

Additionally, to help keep our documentation consistent and make it easy for all users to reference, we provide templates for how-tos and tutorials. You can download these and open them for editing as Jupyter notebooks, and follow [this tutorial]() to prepare your document for submission.

## Submitting your Pull Request

Once you have written your document, it's time to open a Pull Request (PR) to [napari's main repository](https://github.com/napari/napari). If you're not familiar with Git or Pull Requests, follow the steps below to open your Pull Request online through GitHub. If you already know how to submit Pull Requests but aren't sure where to put your document, go to [Step ##]().

### 1. Fork the napari repository

Signed into your GitHub account, go to the [napari repository](https://github.com/napari/napari) and fork it to your own account by clicking the `Fork` button in the top right.

![Screenshot of top right of napari's GitHub page with an arrow pointing to the Fork button](./fork_repo.png)

### 2. Navigate to the right folder

Click on folder names within the repository to navigate to the folder where your document needs to go.

+++

````{admonition} **Tutorials**
:class: dropdown

Path to tutorials folder here

````


````{admonition} **How-Tos**
:class: dropdown

Path to how-tos here

````


````{admonition} **Explanations**
:class: dropdown

Path to explanations here

````

+++

### 3. Upload your file

Once you're in the right folder, you can click on `Add File -> Upload Files`, then drag your file into the box or click `choose your files` and navigate to your document on your computer.

![Screenshot of top right of repository page with an arrow pointing to Add File, Upload Files](./upload_files.png)

+++

### 4. Open the Pull Request
