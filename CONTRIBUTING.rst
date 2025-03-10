Contributing
============

If you want to contribute to ``LenslessPiCam`` and make it better, your
help is very welcome. Contributing is also a great way to learn more
about the package itself.

Ways to contribute
------------------

-  File bug reports, or suggest improvements / feature requests by
   opening an “Issue”.
-  Opening a “Pull request” (see below).

Developer installation
----------------------

.. code:: bash

   # clone repository
   git clone git@github.com:LCAV/LenslessPiCam.git
   cd LenslessPiCam

   # install in virtual environment
   conda create --name lensless python=3.9
   conda activate lensless

   # install library and dependencies
   (lensless) pip install -e .

   # run an example reconstruction
   python scripts/recon/admm.py

   # run unit tests
   (lensless) pip install pytest pycsou
   (lensless) pytest test/

   # additional requirements for reconstructions and metrics
   # separated due to heavy installs for `lpips` and `pycsou` 
   # which may not be needed on the Raspberry Pi
   (lensless) pip install -r recon_requirements.txt

Coding style
------------

We use `Black <https://github.com/psf/black>`__ and
`Flake8 <https://flake8.pycqa.org/en/latest/>`__ for code-formatting.

We recommend setting up pre-hooks to check that you meet the style
guide:

.. code:: bash

   # install inside virtual environment
   (lensless) pip install pre-commit black

   # Install git hooks in `.git/` directory
   (lensless) pre-commit install

When you do your first commit, the environments for Black and Flake8
will be initialized.

You can manually run Black with the provided script:

.. code:: bash

   ./format_code.sh

Unit tests
----------

As much as possible, it is good practice to write unit tests to make
sure code does not break when adding new functionality. We have prepared
a few which can be run with `pytest <https://docs.pytest.org>`__.

First install the library:

::

   pip install pytest

And then run

::

   pytest test/

to run all tests.

How to make a clean pull request (PR)
-------------------------------------

1. Create a personal fork of the project.
2. Clone the fork on your local machine.
3. Create a new branch to work on. Give it a new name that reflects the bug /
   feature you will work on. It is recommended to keep the main branch
   “clean” and in sync with the original repository's main branch!
4. Implement / fix your feature, comment your code.
5. Write or adapt tests as needed.
6. Add or change the documentation as needed.
7. Format the code (see above).
8. Push your new branch to your fork.
9. Open a pull request with the original repository.

Release new version and deploy to PyPi
--------------------------------------

After merging to the ``main`` branch and from the ``main`` branch (!):

1. Edit the ``lensless/version.py`` file.
2. Update ``CHANGELOG.rst`` with new release version, and create a new 
   section for ``Unreleased``.
3. Commit and push new version to GitHub.

   .. code:: bash

      git add lensless/version.py CHANGELOG.rst
      git commit -m "Bump version to vX.X.X."
      git push origin main

4. Create new tag. 

   .. code:: bash

      git tag -a vX.X.X -m "Description."
      git push origin vX.X.X

5. Create package and upload to Pypi (``pip install twine`` if not
   already done).

   .. code:: bash

      python setup.py sdist
      python -m twine upload  dist/lensless-X.X.X.tar.gz

6. On `GitHub <https://github.com/LCAV/LenslessPiCam/tags>`__ set the
   new tag by (1) clicking "…" and selecting "Create release" and (2) at
   the bottom pressing "Publish release".


Building documentation
----------------------

.. code:: bash

   # create virtual environment
   conda create --name lensless_docs39 python=3.9
   conda activate lensless_docs39

   # install dependencies
   (lensless_docs39) pip install -r docs/requirements.txt

   # build documentation
   (lensless_docs39) python setup.py build_sphinx
   # or
   (lensless_docs39) (cd docs && make html)
   
To rebuild the documentation from scratch:

.. code:: bash

   (lensless_docs39) python setup.py build_sphinx -E -a
