import os
import uuid

import cv2
import pandas as pd

def crop_image(img):
    mask = img!=255
    mask = mask.any(2)
    mask0,mask1 = mask.any(0),mask.any(1)
    return img[np.ix_(mask1,mask0)]

def crop_and_validate(img_path):
    img = cv2.imread(img_path)
    crop_img = crop_image(img)
    if (crop_img.shape[0] == 0) or (crop_img.shape[1] == 0):
        return None
    else:
        return crop_img

def validata_df(df, root_data_dir, output_dir):
    data = df.sample(n=1)
    output = crop_and_validate(os.path.join(root_data_dir, data.image_path))
    if output:
        file_name = str(uuid.uuid4())
        cv2.imwrite(os.path.join(output_dir, f"{file_name}.png"), output)
        origin_text_file_name = data.image_path.split('.')[0] + '.txt'
        with open(os.path.join(root_data_dir, origin_text_file_name), 'r',
                  encoding="utf-8") as f:
            text = f.read()
        with open(f"{file_name}.txt", 'w') as f:
            f.write(text)
        return file_name, text
    else:
        return None

def run(
    catalog_path, root_data_dir, output_data_dir,
    font_family, font_size, sample_size, output_catalog_name
):
    df = pd.read_csv(
        catalog_path, encoding='utf-8'
    )

    sub_df = df[(df.font_family == font_family) & (df.font_size == font_size)]

    output_dir = os.path.join(root_data_dir, output_data_dir)
    os.makedirs(output_dir, exist_ok=True)

    valid_count = 0
    pair_dict = {}
    while valid_count < sample_size:
        output = validata_df(sub_df, root_data_dir, output_dir)
        if output:
            pair_dict[output[0]] = output[1]
            valid_count += 1

    new_catalog = pd.DataFrame({
        'text_path': [k + '.txt' for k in pair_dict.keys()],
        'text': list(pair_dict.values()),
        'image_path': [k + '.png' for k in pair_dict.keys()]
    })

    new_catalog.to_csv(output_catalog_name+'.csv', encoding='utf-8-sig', index=False)



if __name__ == '__main__':
    run()