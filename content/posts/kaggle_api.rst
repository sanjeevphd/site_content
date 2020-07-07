---
title: "Guide to Kaggle API"
weight: 3
tags: ["Kaggle", "CLI"]
---

A quick guide to install and use the official Kaggle API, mostly for personal reference.

Installation
============

Get the `official Kaggle API <https://github.com/Kaggle/kaggle-api>`_.

.. note::
   Kaggle API does not work with Python 2. Only Python 3 is supported.

A recommended first step is to create a separate virtual environment using the desired environment management tool. In Conda, `conda create -n kaggle` does the job. Next, activate the virtual environment. Again, in Conda, `conda activte kaggle` will get this done. Next, decision time - to `conda` or `pip`? Whenever possible I prefer to `conda install` packages. So, `conda install kaggle -c conda-forge` will get `kaggle` installed.

Use `kaggle --version` to verify the installation succeeded.

Usage
=====

.. note::
   Before using the Kaggle API, be sure to obtain the required API credentials by,

   #. Creating an account on the Kaggle site
   #. Navigate to Account and select 'Create API Token'
   #. Save the resulting `kaggle.json` file to the directory `~/.kaggle` (on \*nix)
   #. For Windows specific instructions or for to use environment variable to setup Kaggle credentials (no local file needed) `see here <https://github.com/Kaggle/kaggle-api#api-credentials>`_
   #. To avoid the `Warning: Your Kaggle API key is readable by other users on this system!` message (ack!), perform a `chmod 600 ~/.kaggle/kaggle.json`, which is also included as a part of the warning message, like a good warning message should.
  
Assuming the Kaggle API credentials are in place, you are now ready to start using the command line tool. While listing and general *browsing* of competitions and files are okay, any meaningful interaction requires you to have accepted the competition rules and regulations, which unfortunately can **only** be done via a browser.

**Listing Competitions**

.. code:: 
   :class: bash

   ~ >>kaggle competitions list
   ref                                            deadline             category            reward  teamCount  userHasEntered  
   ---------------------------------------------  -------------------  ---------------  ---------  ---------  --------------  
   digit-recognizer                               2030-01-01 00:00:00  Getting Started  Knowledge       2486           False  
   titanic                                        2030-01-01 00:00:00  Getting Started  Knowledge      13429            True  
   house-prices-advanced-regression-techniques    2030-01-01 00:00:00  Getting Started  Knowledge       4917            True  
   imagenet-object-localization-challenge         2029-12-31 07:00:00  Research         Knowledge         56           False  
   tensorflow2-question-answering                 2020-01-22 23:59:00  Featured           $50,000        177           False  
   data-science-bowl-2019                         2020-01-22 23:59:00  Featured          $160,000        386           False  
   pku-autonomous-driving                         2020-01-21 23:59:00  Featured           $25,000        102           False  
   competitive-data-science-predict-future-sales  2019-12-31 23:59:00  Playground           Kudos       4658           False  

The Kaggle API provides 4 main commands, each with several sub-commands.

- competitions (c)
- datasets (d)
- kernels (k)
- config

It can get tedious to type the lengthy commands, along with the sub-commands and options. Kaggle API provides simple *shorthands* for three of the main commands that I highly recommend using.

**Download Competition Files**

As an example, let's look at the `ASHRAE - Great Energy Predictor III <https://www.kaggle.com/c/ashrae-energy-prediction/overview>`_ competition.

.. code::
   :class: bash

   ~ >>kaggle c list -s ashrae
   ref                       deadline             category   reward  teamCount  userHasEntered  
   ------------------------  -------------------  --------  -------  ---------  --------------  
   ashrae-energy-prediction  2019-12-19 23:59:00  Featured  $25,000       1376    True  

   ~ >>kaggle c files ashrae-energy-prediction
   name                    size  creationDate         
   ---------------------  -----  -------------------  
   test.csv                 1GB  2019-10-10 17:13:34  
   weather_test.csv        14MB  2019-10-10 17:13:34  
   train.csv              647MB  2019-10-10 17:13:34  
   weather_train.csv        7MB  2019-10-10 17:13:34  
   sample_submission.csv  427MB  2019-10-10 17:13:34  
   building_metadata.csv   44KB  2019-10-10 17:13:34  

   ~ >>kaggle c download ashrae-energy-prediction
   ~ >>ls
   ashrae-energy-prediction.zip
   ~ >>unzip ashrae-energy-prediction.zip
   Archive:  ashrae-energy-prediction.zip
     inflating: building_metadata.csv   
     inflating: sample_submission.csv   
     inflating: test.csv                
     inflating: train.csv               
     inflating: weather_test.csv        
     inflating: weather_train.csv       

The downloaded files include a `sample_submission.csv` file. Let's try submitting that file, as is, as our first entry (cause why not?!). Since the inclued sample submission file is a large file, it's better to compress it first (Kaggle API supports zipped/archived file format for submission).

.. code::
   :class: bash

   ~ >>zip sample_submission sample_submission.csv 
   ~ >>kaggle c submit -f sample_submission.csv -c
     adding: sample_submission.csv (deflated 79%)
   ~ >>kaggle c submit ashrae-energy-prediction -f sample_submission.zip -m "Testing Kaggle API submission with dummy file" 
   100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 89.1M/89.1M [02:42<00:00, 573kB/s]
   Successfully submitted to ASHRAE - Great Energy Predictor III%                                                                                       

Let's check our latest submission.

.. code::
   :class: bash

   ~ >>kaggle c submissions -c ashrae-energy-prediction
   fileName               date                 description                                    status    publicScore  privateScore  
   ---------------------  -------------------  ---------------------------------------------  --------  -----------  ------------  
   sample_submission.zip  2019-10-31 16:43:41  Testing Kaggle API submission with dummy file  complete  4.69         None          

How did we do? Let's check the leaderboard for what the top scores are.

.. code::
   :class: bash

   ~ >>kaggle c leaderboard ashrae-energy-prediction -s
   teamId  teamName                 submissionDate       score  
   -------  -----------------------  -------------------  -----  
   3799768  STL                      2019-10-31 17:15:02  1.07   
   3800952  FabienDaniel             2019-10-29 21:17:25  1.07   
   3796602  eagle4                   2019-10-31 17:44:23  1.07   
   3801546  Oleg Knaub               2019-10-31 15:01:36  1.08   
   3795818  Vicens Gaitan            2019-10-31 09:42:05  1.08   

In addition the Kaggle API supports browsing, downloading, creating `datasets <https://github.com/Kaggle/kaggle-api#datasets>`_, browsing, pushing and pulling `kernels <https://github.com/Kaggle/kaggle-api#kernels>`_ and API `configuration <https://github.com/Kaggle/kaggle-api#config>`_ that makes it easy to use from the command line.
