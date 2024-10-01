
import os


folder_path = "/home/ilias.papastratis/Downloads/toyotav2/img/1_t0y0tamr2 car/"
folder_path = "/home/ilias.papastratis/Downloads/toyota/"
# Use os.listdir to get all files in the directory
files_in_dir = os.listdir(folder_path)

# Filter out any non-image files
image_files = [file for file in files_in_dir if file.endswith(('jpg', 'png', 'jpeg'))]

prompts = [
    " parked in a asphalt road  sprawling metropolis at sunset with skyscrapers and busy streets, ",
    " parked in an asphalt road in front of  a  tranquil countryside landscape with rolling hills and clear blue skies.",
    " parked in an asphalt road in front of  old city buildings.",
    " parked in a road with a  cityscape with high-tech buildings and neon lights.",
    " parked in an asphalt road in front of a  mountainous landscape with pine trees.",
    " parked in an asphalt road in front of a  coastal city with palm trees ",
    " parked in an asphalt road in front of a  dense forest landscape with a narrow path and lush greenery.",
    " parked in an asphalt road on a  city with  traditional houses.",
    " parked in an asphalt road in front of   a  snowy cityscape with tall buildings and snow-covered streets.",
    " parked in an asphalt road in front of  a  tropical island city with colorful houses and crystal-clear waters.",
    " parked in an asphalt road in front of a  panoramic view of a peaceful valley with a river flowing through it.",
    " parked in an asphalt road in front of a  breathtaking view of a majestic mountain range under a clear sky.",
    " parked in an asphalt road in front of a  serene lake surrounded by dense forests and wildflowers.",
    " parked in an asphalt road in front of a  mesmerizing sunset over a calm ocean with gentle waves.",
    " parked in an asphalt road in front of a  vibrant autumn landscape with trees full of colorful leaves.",
    " parked in an asphalt road in front of a  captivating view of a desert with towering sand dunes under a scorching sun.",
    " parked in an asphalt road in front of a  picturesque view of a tropical beach with white sands and turquoise waters."
]
#
# for idx,file in enumerate(image_files):
#
#
#     rendered_img = Image.open(os.path.join(folder_path, file)).convert('RGB')
#     generated_img_list = generate(
#         rendered_img,
#         positive_prompt=prompts[idx%len(prompts)],
#         negative_prompt="bad quality, low resolution",
#         seed=-1,
#         depth_map_feather_threshold=depth_map_feather_threshold,
#         depth_map_dilation_iterations=depth_map_dilation_iterations,
#         depth_map_blur_radius=depth_map_blur_radius,
#         progress=gr.Progress(track_tqdm=True)
#     )
#
#     generated_images = generated_img_list[0]
#     pre_processing_images = generated_img_list[2]
#     for y_idx,new_img in enumerate(generated_images):
#         print(type(new_img))
#         new_path = f"/home/ilias.papastratis/Downloads/toyota_bg3/prompt_{idx%len(prompts)}_{y_idx}_{file}"
#         new_img.save(new_path)
#     #
#     # for y_idx,new_img in enumerate(pre_processing_images):
#     #     print(type(new_img))
#     #     new_path = f"/home/ilias.papastratis/Downloads/toyota_bg3/preprocess{new_img[-1]}_{idx%len(prompts)}_{y_idx}_{file}"
#     #     new_img[0].save(new_path)


manual_captions = {"front":"The image shows a front view a Formula E race car, which is an electric-powered single-seater racing car. The car has a  blue and black metallic color with white logos. The car has a large rear wing and a diffuser. It is parked on a gray surface with a white wall in the background. The car is lit by sunlight coming from the left side of the image. The car has a sleek, aerodynamic design with a low profile and a wide stance. The front of the car has a large air intake and a splitter, while the rear of the car has a diffuser and a rear wing. The car is equipped with Hankook tires and has a halo device above the cockpit for driver protection.",
                   "side":"The image shows a side view of a Formula E race car, which is an electric-powered single-seater racing car. The car is blue and black, with white logos. It has a sleek, aerodynamic design, with a low-slung body and a large rear wing. The car is equipped with Michelin tires and has the following sponsors: ABB, Allianz, Julius Bär, Boss, and Schaeffler.  The front of the car has a large air intake and a splitter, while the rear of the car has a diffuser and a rear wing. The car is equipped with Hankook tires and has a halo device above the cockpit for driver protection. The car is casting a shadow on the ground, which is white. The background is a light gray gradient.",
                   "back": "The image shows a back view of  a Formula E race car from the rear. The car has a  blue and black metallic color with white logos. The car has a large rear wing and a diffuser. The tires are slicks, and the wheels are black with Hankook brand. The car is equipped with Michelin tires and has the following sponsors: ABB, Allianz, Julius Bär, Boss, and Schaeffler.  The front of the car has a large air intake and a splitter, while the rear of the car has a diffuser and a rear wing. The car is equipped with Hankook tires and has a halo device above the cockpit for driver protection. The car is on a gray track. The background is a light gray gradient."}



def manuan_captioning(prompts):
    POSITIVE_PROMPT_PREFIX = "Raw realistic photo,"
    POSITIVE_PROMPT_SUFFIX = "masterpiece,4K,HD,commercial product photography, 24mm lens f/8"
    folder_path = "/home/ilias.papastratis/workdir/data/formula/"
    # Use os.listdir to get all files in the directory
    files_in_dir = os.listdir(folder_path)

    # Filter out any non-image files
    image_files = [file for file in files_in_dir if file.endswith(('jpg', 'png', 'jpeg'))]
    for img_file in image_files:
        print(img_file)
        prompt_idx = int(img_file.split('prompt_')[-1].split('_')[0])
        angle = int(img_file.split('blender_')[-1].split('_')[0])

        print(prompt_idx,angle)
        if angle >= 0 and angle < 31:
            view = 'front'
        elif angle > 31 and angle < (180 - 31):
            view = 'side'
        elif angle > 150 and angle < (180 + 31):
            view = 'back'
        elif angle > 210 and angle < (360 - 31):
            view = 'side'
        elif angle > 320:
            view = 'front'
        # caption = f"{POSITIVE_PROMPT_PREFIX} of {view} view of t0y0tamr2 with camera angle {angle} {prompts[prompt_idx]}"
        # print(caption)
        # caption_file_path = img_file.replace('.png','.txt')
        # print(os.path.join(folder_path,caption_file_path))
        # with open(os.path.join(folder_path,caption_file_path),'w') as f:
        #     f.write(caption)

manual_captions = {"front":"The image shows a front view a Ford Explorer EV suv car that has a light silver blue color.The car is facing the camera and is slightly angled to the left. There is a blue Ford logo in the front mask of the car. The front of the car is in focus, while the back is slightly blurred. The car has a sleek, modern design with sharp lines and a bold front grille. The headlights are narrow and angular, and the taillights are thin and wrap around the sides of the car. The car has a black roof and black accents on the front and rear bumpers. The wheels are large and have a sporty design. The car is parked in a studio setting with a white wall in the background. The lighting is soft and natural, and there are no shadows on the car.",
                   "side":"The image shows a side view of a Ford Explorer EV suv car. The car   has a light silver blue color.  The car is facing the camera at a slight angle, and its headlights and taillights are on. The car has a sleek, modern design, with sharp lines and a sloping roofline. The front of the car features a large grille with the Ford logo in the center. The headlights are narrow and angular, and they are connected by a thin strip of LED lights. The car has a black roof and black side mirrors. The wheels are large and have a multi-spoke design. The background of the image is a light blue color, and there is a white wall in the distance.",
                   "back": "The image shows a back view of  Ford Explorer EV suv car. The car   has a light silver blue color. The car is parked on a light blue surface with a white background. The car has a sleek and modern design, with sharp lines and a sporty stance. The rear of the car features a large spoiler, a black diffuser, and LED taillights. The car also has a panoramic sunroof and black roof rails. The wheels are black and silver, and the tires are low-profile. The car is badged with the Ford logo on the rear hatch and the name of the car EXPLORER with black color."}

def manual_captioning_blender_images( ):
    POSITIVE_PROMPT_PREFIX = "Raw realistic photo,"
    POSITIVE_PROMPT_SUFFIX = "masterpiece,4K,HD,commercial product photography, 24mm lens f/8"
    folder_path = "/home/ilias.papastratis/workdir/data/ford_3d/img/5_fordexplorerev car/"
    # Use os.listdir to get all files in the directory
    files_in_dir = os.listdir(folder_path)

    # Filter out any non-image files
    image_files = [file for file in files_in_dir if file.endswith(('jpg', 'png', 'jpeg'))]
    for img_file in image_files:
        print(img_file)

        angle =  img_file.split('blender_')[-1].split('_')[0]

        print(  angle)
        caption = manual_captions[angle]

        # print(caption)
        caption_file_path = img_file.replace('.png','.txt')
        print(os.path.join(folder_path,caption_file_path))
        with open(os.path.join(folder_path,caption_file_path),'w') as f:

            f.write(caption)
        f.close()

def fix_angle_blender_images():
    POSITIVE_PROMPT_PREFIX = "Raw realistic photo,"
    POSITIVE_PROMPT_SUFFIX = "masterpiece,4K,HD,commercial product photography, 24mm lens f/8"
    folder_path = "/home/ilias.papastratis/workdir/data/ford_3d/img/5_fordexplorerev car/"
    # Use os.listdir to get all files in the directory
    files_in_dir = os.listdir(folder_path)

    # Filter out any non-image files
    image_files = [file for file in files_in_dir if file.endswith(('jpg', 'png', 'jpeg'))]
    for img_file in image_files:
        print(img_file)

        view = img_file.split('blender_')[-1].split('_')[0]
        angle = int(img_file.split('blender_')[-1].split('_')[1].split('_')[0])
        print(view,angle)
        if angle >= 0 and angle < 29:
            nview = 'front'
        elif angle > 29 and angle < 120:
            nview = 'side'
        elif angle > 120 and angle < 191:
            nview = 'back'
        elif angle > 191 and angle < 331:
            nview = 'side'
        elif angle > 331:
            nview = 'front'

        if view!=nview:
            new_img_file = img_file.replace(view, nview)
            print('FIXED!!!!!!!!!!!!!!!!!!!')
            print(os.path.join(folder_path,img_file))
            print(os.path.join(folder_path,new_img_file))
            os.rename(os.path.join(folder_path,img_file),os.path.join(folder_path,new_img_file))
