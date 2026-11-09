from ultralytics import YOLO
from PIL import Image

# Define path to directories containing images and videos for inference
source_img_path = "examples/input/images"
images_categories = [
    "same",         # same accuracy of bird detection 
    "better",       # better accuracy of bird detection after fine-tuning the model
    "some_cases",   # some interesting cases of bird detection
    "aircraft",     # some aircraft images to test the model's ability to distinguish birds from other objects before\after fine-tuning
    "other_animals", # human, cows, bats and insects to test the model's ability to distinguish birds from other flying animals before\after fine-tuning
    #"new"
]

weights_path = 'weights/'
weights_list = ["yolo26n.pt", "birds_detect_best_01.pt"]
for weights_name in weights_list:
    # Run inference on the sources
    model = YOLO(weights_path + weights_name) # load a pretrained YOLO26n model
    for image_category in images_categories:
        source_img_dir = f"{source_img_path}/{image_category}/"
        print(f"Running inference on {source_img_dir} using model {weights_name}")
        results_img = model(source_img_dir, stream=True)  # generator of Results objects
        # Visualize the results
        for i, r in enumerate(results_img):
            # Plot results image
            im_bgr = r.plot()  # BGR-order numpy array
            im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
            #r.show() # Show results to screen (in supported environments)
            # Save results to disk
            i_str = f"0{i}" if i < 10 else str(i)
            r.save(filename=f"examples/output/images/{image_category}/bird_detection_test_{i_str}_{weights_name}.jpg")

