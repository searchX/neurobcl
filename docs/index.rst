.. NeuroBCL documentation master file, created by
   sphinx-quickstart on Wed Feb 21 01:45:33 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NeuroBCL's documentation!
====================================
NeuroBCL (Neuro Bucket Classifier) is a Python package for finding ranges for numeric features normalized 
with percentile distribution, the basic idea is to divide the range of the feature into buckets and then 
classify the data into these buckets. 

Installation
------------

[Coming Soon] Install the ``neurobcl`` package with `pip
<https://pypi.org/project/neurobcl>`_:

.. code-block:: console

    $ pip install neurobcl

Or install the latest package directly from github

.. code-block:: console

    $ pip install git+https://github.com/searchX/neurobcl

Example Usage
-------------
A simple example of using the package is as follows:

Index classifier on our sample data below:

.. code-block:: python

   from neurobcl.main import train_from_dictionary
   classifier = train_from_dictionary([
            {"color": "red", "size": "small", "price": 100},
            {"color": "blue", "size": "small", "price": 200},
            {"color": "red", "size": "large", "price": 300},
            {"color": "blue", "size": "large", "price": 400},
        ], ["color", "size"], ["price"])

1. Minimum price that should be greater than for it to be in bucket 1 atleast

.. code-block:: python

   classifier.get("price", 1, '>')
   # Output: 100

2. This is the price that will be the limit of all items that can exist until bucket 4

.. code-block:: python

   classifier.get("price", 4, '<')
   # Output: 400

3. Use filter, to get the items that are in bucket 1 and color blue

.. code-block:: python

   classifier.get("price", 1, '>', filters={"color": "blue"})
   # Output: 200

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   neurobcl

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
