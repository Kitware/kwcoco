The kwcoco CLI uses `argcomplete <https://pypi.org/project/argcomplete/>`_.
To gain the tab-complete feature in bash run:

.. code-block:: bash

    pip install argcomplete
    mkdir -p ~/.bash_completion.d
    activate-global-python-argcomplete --dest ~/.bash_completion.d
    source ~/.bash_completion.d/python-argcomplete

Then to gain the feature in new shells put the following lines in your .bashrc:

.. code-block:: bash

    if [ -f "$HOME/.bash_completion.d/python-argcomplete" ]; then
        source ~/.bash_completion.d/python-argcomplete
    fi

If you know of a way to have this feature "install itself" or avoid requiring
this manual step, please submit an MR!
