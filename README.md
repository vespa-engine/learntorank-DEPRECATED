<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

Vespa for Data Scientists
================

![](https://vespa.ai/assets/vespa-logo-color.png)

See documentation at [vespa-engine.github.io/learntorank/](https://vespa-engine.github.io/learntorank/)



## Motivation

This library contains application specific code related to data
manipulation and analysis of different Vespa use cases.
The [Vespa python API](https://pyvespa.readthedocs.io/) is used to interact with
Vespa applications from python for faster exploration.

The main goal of this space is to facilitate prototyping and experimentation for data scientists.
Please visit Vespa [sample apps](https://github.com/vespa-engine/sample-apps/)
for production-ready use cases and [Vespa docs](https://docs.vespa.ai/) for in-depth Vespa documentation.



## Install

Code to support and reproduce the use cases documented here can be found in the `learntorank` library.

Install via PyPI:

`pip install learntorank`



## Development

All the code and content of this repo is created using [nbdev](https://nbdev.fast.ai/) by editing notebooks.
We will give a summary below about the main points required to contribute,
but we suggest going through [nbdev tutorials](https://nbdev.fast.ai/tutorials/tutorial.html) to learn more.


### Setting up environment

1. Create and activate a virtual environment of your choice.
    We recommend [pipenv](https://github.com/pypa/pipenv).

    ``` bash
    pipenv shell
    ```

3. Install Jupyter Lab (or Jupyter Notebook if you prefer).

    ``` bash
    pip3 install jupyterlab
    ```

4. Create a new kernel for Jupyter that uses the virtual environment created at step 1.

    -   Check where the current list of kernels is located with
        `jupyter kernelspec list`.
    -   Copy one of the existing folder and rename it to `learntorank`.
    -   Modify the `kernel.json` file that is inside the new folder to
        reflect the `python3`executable associated with your virtual
        env.

5. Install `nbdev` library:

    ``` bash
    pip3 install nbdev
    ```

6. Install `learntorank` in development mode:

    ``` bash
    pip3 install -e .[dev]
    ```


### Most used nbdev commands

From your terminal:

-   `nbdev_help`: List all nbdev commands available.

-   `nbdev_readme`: Update `README.md` based on `index.ipynb`

-   Preview documentation while editing the notebooks:

    -   `nbdev_preview --port 3000`

-   Workflow before pushing code:

    -   `nbdev_test --n_workers 2`: Execute all the tests inside
        notebooks.
        -   Tests can run in parallel but since we create Docker
            containers we suggest a low number of workers to preserve
            memory.
    -   `nbdev_export`: Export code from notebooks to the python
        library.
    -   `nbdev_clean`: Clean notebooks to avoid merge conflicts.

-   Publish library

    -   `nbdev_bump_version`: Bump library version.
    -   `nbdev_pypi`: Publish library to PyPI.

----

[![/integration tests](https://cd.screwdriver.cd/pipelines/10949/tests/badge)](https://cd.screwdriver.cd/pipelines/10949)
