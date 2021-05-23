import io.read_image_file as rif


image_params = [['../data/raw_data/d21 - KFC,McDonald_rec_I.zip',
                 'jpg', (128, 128), 'd21', 100],
                ['../data/raw_data/d20 - hand_gestures_I.zip',
                 'jpg', (128, 128), 'd20', 100],
                ['../data/raw_data/d18 - blood cells_I.zip',
                 'jpeg', (128, 128), 'd18', 100],
                ['../data/raw_data/d17 - gemstones_I.zip',
                 'jpg', (128, 128), 'd17', 100],
                ['../data/raw_data/d5 - food_rec_I.zip',
                 'jpg', (128, 128), 'd5', 100],
                ['../data/raw_data/d4 - flower_rec_i.zip',
                 'png', (128, 128), 'd4', 100],
                ['../data/raw_data/d3 - flower_rec_I.zip',
                 'jpg', (128, 128), 'd3', 100],
                ['../data/raw_data/d1 - frut_rec_I.zip',
                 'jpg', (128, 128), 'd1', 100]
                ]


def read_images(number_of_records=None):

    data_images = []

    for param in image_params:
        data_images.append(rif.read_images(
            file_location=param[0], file_format=param[1], image_size=param[2], set_name=param[3], num_of_records=param[4]))

    return data_images
