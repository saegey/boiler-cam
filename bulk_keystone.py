with open("points.json", "r") as file:
    saved_points = json.load(file)

# Use the points to correct another image
another_image = cv2.imread("path/to/another/image.jpg")
correct_keystone(another_image, saved_points, save_output_path="another_corrected_image.jpg")
