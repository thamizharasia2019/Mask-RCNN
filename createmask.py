import json

plant_book_mask_image = Image.open('/path/to/images/plant_book_mask.png')
bottle_book_mask_image = Image.open('/path/to/images/bottle_book_mask.png')

mask_images = [plant_book_mask_image, bottle_book_mask_image]

# Define which colors match which categories in the images
houseplant_id, book_id, bottle_id, lamp_id = [1, 2, 3, 4]
category_ids = {
    1: {
        '(0, 255, 0)': houseplant_id,
        '(0, 0, 255)': book_id,
    },
    2: {
        '(255, 255, 0)': bottle_id,
        '(255, 0, 128)': book_id,
        '(255, 100, 0)': lamp_id,
    }
}

is_crowd = 0

# These ids will be automatically increased as we go
annotation_id = 1
image_id = 1

# Create the annotations
annotations = []
for mask_image in mask_images:
    sub_masks = create_sub_masks(mask_image)
    for color, sub_mask in sub_masks.items():
        category_id = category_ids[image_id][color]
        annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
        annotations.append(annotation)
        annotation_id += 1
    image_id += 1

print(json.dumps(annotations))
