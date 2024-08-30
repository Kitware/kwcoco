Mini JQ Tutorial
----------------

The ``jq`` tool is very helpful in navigating json files, and some of the
kwcoco commands will use its query language, so it is a good idea to become
familiar with it. The following bash code generates a demo kwcoco json file and
uses it to show the json data structure.

.. code:: bash

    # Generate toydata
    kwcoco toydata vidshapes-msi-multispectral --bundle_dpath=./my_demo_bundle

    # Call jq on the generated json file, this will print the entire json file in a
    # mostly readable format (although it is a bit long)
    jq '.' ./my_demo_bundle/data.kwcoco.json


    # You can use JQ to only list paths in the nested structure of the json file
    # List only the category table
    jq '.categories' ./my_demo_bundle/data.kwcoco.json

    # List only the image table
    jq '.images' ./my_demo_bundle/data.kwcoco.json

    # List only the first image in the image table
    jq '.images[0]' ./my_demo_bundle/data.kwcoco.json
