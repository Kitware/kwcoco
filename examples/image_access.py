import kwcoco
coco = kwcoco.CocoDataset.demo('vidshapes', num_frames=10, num_videos=1, verbose=0)
# The "images" method provides a "vectorized" view into multiple images
# In this case we have 10 of them
all_images = coco.images()
print(all_images)


# We can grab properties of all items (like selecting a column from a data frame)
all_images.lookup('file_name')


# Lets grab the ID of a single image
image_id: int = all_images.lookup('id')[0]
# The vectorized interface is a convenience, if you want raw access to the
# image dictionaries use the index to lookup information by id
img : dict = coco.index.imgs[image_id]
import ubelt as ub
print(f'img = {ub.urepr(img, nl=1)}')


# The index stores the raw image dictionary, but there is also a CocoImage
# helper class that provides methods on top of the image dictionary data
# you can create this object as follows:
coco_image: kwcoco.CocoImage = coco.coco_image(image_id)
print(coco_image)

# Note: this object stores a reference the raw dictionary
print(f'coco_image.img = {ub.urepr(coco_image.img, nl=1)}')

# This provides a lot of nice helper functions including a way to get the
# vectorized annotations that belong to this image
annots = coco_image.annots()
print(annots)


# But you can also grab this information without relying on any additional
# object construction. The main data structure is the gid-to-aids map:
annot_ids: list[int] = coco.index.gid_to_aids[img['id']]


# Another alternative is to use the arguments to the `CocoDataset.annots` method
annots = coco.annots(image_id=img['id'])



from kwcoco.compat_dataset import COCO
compat_coco = COCO(coco.dataset)
id_ = 1
compat_coco.getAnnIds([id_] )
