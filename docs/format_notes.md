# The Kitware Dataset Format


Keep using kwcoco, or pick a new name?

    * KW-ADF Agnostic Database Format

    * KW-DF Dataset Format

    * KW O D F - Organized Dataset Format

    * KW C O C O - Common Objects in COntext

    * KW C O D E C - Categorized organized dataset exchange 

    * KW C O D A - Categorized organized dataset exchange 

    * KW C I A - Categories Image Annotations

    * KW C A C A O - Common Annotated Categories Archived Observations  

    * KW C A D - Common Annotated Dataset  

    * KW C A G - Common Annotated imaGes  

* Folder based. Each dataset is a folder. 
  The name of the folder is typically the name of the dataset.

The file structure looks like this:



```

    + <ROOT_DPATH>  # the name of this folder should be the name of the dataset
    |
    +---- main.kwcoco.json   # Manifest of all information 
    |
    +---- < this directory can contain any mirror of the main mainfest file 
    |       for example, an SQLite read-only view >
    |    
    +---- .assets  # This directory is where raw data should be stored.
    |    |
    |    +---- images
    |    |     |
    |    |     +---- <arbitrary folder of image>
    |    |      
    |    +---- videos
    |    |     |
    |    |     +---- <arbitrary folder videos>
    |    |     
    |    +---- < The manifest can reference any file in this directory as needed by the dataset itself >
    |
    +---- .cache
         |
         +---- < arbitrary cache directory for applications to use, this can
                 store things like a cog cache, frames extracted from the 
                 videos, or recomputable features >


```
