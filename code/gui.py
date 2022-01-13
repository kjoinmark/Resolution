#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os.path

import PySimpleGUI as sg
import cv2 as cv

import resolution

# First the window layout in 2 columns
sg.theme('Reddit')

file_types = [("All files (*.*)", "*.*"),
              ("JPEG (*.jpg)", "*.jpg")]

file_list_column = [
    [sg.Image(key="-IMAGE-")],
]


# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Input(size=(50, 2), key="-FILE-"),
     sg.FileBrowse(file_types=file_types)],
    # [sg.Button("Обработать изображение")],
    [sg.Button("Загрузить изображение")],
    [sg.Button("Найти разрешение")],
    [sg.Text("Выберите порог")],
    [sg.InputText(size=(15, 1), key='thresh_slider')],
    [sg.Text("Разрешение:")],
    [sg.Text(size=(20, 1), key="-TOUT2-")],
    [sg.Text(size=(40, 2), key="-TOUT3-")]
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Разрешение", layout, margins=(0, 60), element_justification='center', resizable=True).Finalize()
window.Maximize()

is_loaded = False
is_preprocessed = False
res_preprocessed = False

window_w, window_h = sg.Window.get_screen_size()


def resize_img(img, w_h=window_h):
    w, h = img.shape[::-1]
    scale = w_h * 0.8 / h
    resized_image = cv.resize(img, (int(w * scale), int(h * scale)))
    return resized_image


# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == "Загрузить изображение":
        window_w, window_h = window.size
        filename = values["-FILE-"]
        #print(filename)
        if os.path.exists(filename):
            res = resolution.Resolution(values["-FILE-"])
            image = resize_img(res.image, window_h)
            imgbytes = cv.imencode('.png', image)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)
            is_loaded = True
            is_preprocessed = False
            window["-TOUT3-"].update('')
        # if event == "Обработать изображение":
        if is_loaded and not is_preprocessed:
            res.align_image()
            res_preprocessed = res.pre_processing()
            image = resize_img(res.image, window_h)
            imgbytes = cv.imencode('.png', image)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)

            is_preprocessed = True
            if res_preprocessed:
                str = 'Ошибка при обработке, возможен неверный результат'
                window["-TOUT3-"].update(str)
                sg.popup(str)

    # if event == "Сохранить для датасета":
    #   if is_preprocessed:
    #      res.save_for_ds()

    if event == "Найти разрешение":
        if is_preprocessed:
            if len(values['thresh_slider']) == 0:
                threshold = 0.5
            else:
                threshold = float(values['thresh_slider'])

            threshold_h, threshold_v = res.find_resolution_network(threshold)
            image = resize_img(res.image, window_h)
            imgbytes = cv.imencode('.png', image)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)
            window["-TOUT2-"].update(
                '%5.1f' % ((res.get_table_resolution(res.vertical_resolution) + res.get_table_resolution(
                    res.horizontal_resolution)) / 2))
          

window.close()
