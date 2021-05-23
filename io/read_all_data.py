import read_image_file as lib


def read_images(number_of_records=None):

    data_images = []

    for param in params:
        date_images.append(lib.read_images(
            file_location=param[0], file_format=param[1], image_size=param[2], set_name=param[3], num_of_records=param[4]))

    return data_images


nr_of_set = 2

params = [['../data/raw_data/d21 - KFC,McDonald_rec_I.zip',
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


#lib.read_images(file_location='../data/raw_data/d17 - gemstones_I.zip', file_format='jpg', image_size=(128, 128), set_name='d17', num_of_records=10)
lib.read_images(file_location=params[nr_of_set][0],
                file_format=params[nr_of_set][1],
                image_size=params[nr_of_set][2],
                set_name=params[nr_of_set][3],
                num_of_records=params[nr_of_set][4])




