import os
import keras
import numpy as np
from PIL import Image

if __name__ == '__main__':
    model = keras.models.load_model('./gmike19_model.h5')
    path_to_data = './DATA'
    filenames = [f for f in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, f))]
    number_of_channels = 1
    image_array = []
    filenames_without_ext = []
    for filename in filenames:
        image = Image.open(path_to_data + '/' + filename).convert('L').resize((45, 45))
        image = np.array(image.getdata(), dtype=np.uint8).reshape(image.size[0], image.size[1], number_of_channels)
        image_array.append(image)
        filenames_without_ext.append(os.path.splitext(filename)[0])
    image_array = np.stack(image_array)
    image_array = image_array / 255.0
    results = model(image_array)
    predicted_letter_nums = []
    for res in results:
        predicted_letter_num = np.argmax(res) + 1
        predicted_letter_nums.append(predicted_letter_num)
    filename_letter_pairs = list(zip(filenames_without_ext, predicted_letter_nums))
    output_ready_list = sorted([p[0] + '-' + str(p[1])for p in filename_letter_pairs])
    txtfile = open("gmike19.txt", "w")
    for el in output_ready_list:
        txtfile.write(el + '\n')
    txtfile.close()
