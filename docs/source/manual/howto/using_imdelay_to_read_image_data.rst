Reading Image Data With ImDelay
-------------------------------


.. code:: python

   import kwcoco
   # Given some coco dataset (put your own here)
   coco_dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')

   # Grab the first image id
   image_id = list(coco_dset.images())[0]

   # Then get the CocoImage object:
   coco_img = coco_dset.coco_image(image_id)

   # This object gives you access to the underlying image data.
   # This call just gives you a view into it, but does not do any real IO.
   delayed_image = coco_img.imdelay()

   # You can list what channels you have:
   channels = delayed_image.channels

   # Lets say you want the last 4 channels
   # This may be different in your case.
   channels_of_interest = channels[-4:]

   # Use "take_channels" to only grab the ones you are interested in.
   sub_delayed = delayed_image.take_channels(channels_of_interest)

   # Finalize to load the actual image data into memory
   actual_data = sub_delayed.finalize()
