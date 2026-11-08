from PIL import Image


def crop_image(image: Image.Image, bbox: list):
    """Crop the image based on the bounding box coordinates to zoom in on
    specific regions for detailed analysis.

    Args:
        image: The input image.
        bbox: Normalized bounding box [x1, y1, x2, y2], where
            0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0. (x1,y1) is
            top-left corner, (x2,y2) is bottom-right corner.
    """

    # Validate bbox structure
    if len(bbox) != 4:
        raise Exception(f"Invalid bbox format: {bbox}")

    img_x, img_y = image.size

    # parse bbox and validate
    x1, y1, x2, y2 = [float(coord) for coord in bbox]

    if not (0 <= x1 <= 1.0 and 0 <= y1 <= 1.0 and 0 <= x2 <= 1.0 and 0 <= y2 <= 1.0):
        raise Exception(
            f"In relative mode, all coordinates must be between 0 and 1. Got: {bbox}")

    if not (x1 < x2 and y1 < y2):
        raise Exception(
            f"Invalid bbox coordinates: x1 must be < x2 and y1 must be < y2. Got: {bbox}")

    return image.crop((
        int(x1 * img_x), int(y1 * img_y),
        int(x2 * img_x), int(y2 * img_y)
    ))
