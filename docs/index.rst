Welcome to Music Genre Detection!
=================================

This project implements a CNN-LSTM_Attention model for music genre clasification

Features
--------

- Dataset caching for 6-10x faster training
- Multi-genre clasification (10 genres from GTZAN)
- Mel-spectrogram preprocessing with augmentation
- Professional logging and monitoring

Contents
--------

.. toctree::
    :maxdepth: 2

    modules

Quick Start
-----------
1. Install dependencies:

   .. code-block:: bash

       pip install -r requirements.txt

2. First run (preprocessing):
   .. code-block:: bash

       python main.py

3. Subsequent runs (training/inference):
   .. code-block:: bash

       python main.py

indicies and tables
========================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`