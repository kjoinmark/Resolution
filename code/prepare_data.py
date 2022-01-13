import csv
import glob

import cv2 as cv
import numpy as np


def normalize(pixels):
    pixels = np.asarray(pixels)
    pixels = pixels.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std
    # clip pixel values to [-1,1]
    pixels = np.clip(pixels, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    pixels = (pixels + 1.0) / 2.0
    return pixels

def open_image(filename):
    stream = open(filename, 'rb')
    bytes = bytearray(stream.read())
    array = np.asarray(bytes, dtype=np.uint8)
    image = cv.imdecode(array, cv.IMREAD_GRAYSCALE)
    return image


def save_for_ds(path, filewriter):
    config = np.loadtxt('template_config.txt', dtype=int)
    print(path[10:])
    horizontal = path[path.rfind('Г') + 1:path.rfind('Г') + 4]
    vertical = path[path.rfind('В') + 1:path.rfind('В') + 4]
    horizontal_idx = int(horizontal[0]) * 6 + int(horizontal[-1]) - 1
    vertical_idx = int(vertical[0]) * 6 + int(vertical[-1]) - 1
    print(horizontal, horizontal_idx)
    print(vertical, vertical_idx)
    
    for i in range(12, 33):
       
        filename0 = path + f'\\img_{i}_0.png'
        filename1 = path + f'\\img_{i}_1.png'
        img0 = open_image(filename0)
        img1 = open_image(filename1)
        if config[i, 2] == 0:
            img0 = cv.rotate(img0, cv.ROTATE_90_CLOCKWISE)
            if i <= horizontal_idx:
                filewriter.writerow(np.append(img0, 1).flatten())
            else:
                filewriter.writerow(np.append(img0, 0).flatten())
            if i <= vertical_idx:
                filewriter.writerow(np.append(img1, 1).flatten())
            else:
                filewriter.writerow(np.append(img1, 0).flatten())

        if config[i, 3] == 0:
            img1 = cv.rotate(img1, cv.ROTATE_90_CLOCKWISE)
            if i <= horizontal_idx:
                filewriter.writerow(np.append(img1, 1).flatten())
            else:
                filewriter.writerow(np.append(img1, 0).flatten())
            if i <= vertical_idx:
                filewriter.writerow(np.append(img0, 1).flatten())
            else:
                filewriter.writerow(np.append(img0, 0).flatten())

   

def make_csv():
    w_file = open('csv\\all.csv', mode="w", encoding='utf-8')
    file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
    feature_list = []
    for i in range(115 * 115):
        feature_list.append(f'feature{i}')
    feature_list.append('result')
    file_writer.writerow(feature_list)

    all_ = glob.glob(".\\results\\*")
    print(glob.glob(".\\results\\*"))
    for pack in all_:
        save_for_ds(path=pack, filewriter=file_writer)

    w_file.close()


# save_for_ds(path="results/КО без диафрагмы_Г3-3_В3-3")


all_results = glob.glob(".\\results\\source\\моя оценка\\*")
all_source = glob.glob(".\\source\\моя оценка\\*")
print(glob.glob(".\\results\\source\\моя оценка\\*"))
good = []
everything = []

for file in all_source:
    print(file[file.rfind('\\')+1:])
    everything.append(file[file.rfind('\\')+1:])
string_all = '\n'.join(everything)
print('\n\n')
for file in all_results:
    name = file[file.rfind('\\')+1:]
    if string_all.find(name)!= -1:
        string_all = string_all.replace(name,'')
    print(file[file.rfind('\\')+1:])
    good.append(file[file.rfind('\\')+1:])

print('\nthat all \n ')
print(string_all)

for pack in all_results:
    if pack.find('.ini') != -1:
        continue
    all_files = glob.glob(pack + '\\*.png')
    print(pack)
    images = []
    results = []
    if len(all_files) == 0:
        continue
    for i in range(12, 24):
        #print(i)
        file = glob.glob(pack + f'\\img*{i}*.png')
        for path in file:

            stream = open(path, 'rb')
            bytes = bytearray(stream.read())
            array = np.asarray(bytes, dtype=np.uint8)
            img = cv.imdecode(array, cv.IMREAD_GRAYSCALE)
            images.append(normalize(img))
            res = path[path.rfind('result') + 6:path.rfind('result') + 7]
            # print(file)
            if res != 'N':
                # print(res)
                results.append(int(path[path.rfind('result') + 6:path.rfind('result') + 7]))

  
    pixels = np.asarray(images)
  
    for i, f_img in enumerate(pixels):
        np.savetxt(pack + '\\' + f'img_{i}_{results[i]}.txt', f_img, delimiter=',', fmt='%.10f', encoding='utf-8')
