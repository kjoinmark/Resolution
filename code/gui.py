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

"""
col_names0 = [[sg.Listbox(values=('(0, 1)', '(0, 2)', '(0, 3)', '(0, 4)', '(0, 5)', '(0, 6)'),
                          select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(5, 7), no_scrollbar=True)]]
col_names1 = [[sg.Listbox(values=('(1, 1)', '(1, 2)', '(1, 3)', '(1, 4)', '(1, 5)', '(1, 6)'),
                          select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(5, 7), no_scrollbar=True)]]
col_names2 = [[sg.Listbox(values=('(2, 1)', '(2, 2)', '(2, 3)', '(2, 4)', '(2, 5)', '(2, 6)'),
                          select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(5, 7), no_scrollbar=True)]]
col_names3 = [[sg.Listbox(values=('(3, 1)', '(3, 2)', '(3, 3)', '(3, 4)', '(3, 5)', '(3, 6)'),
                          select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(5, 7), no_scrollbar=True)]]
col_names4 = [[sg.Listbox(values=('(4, 1)', '(4, 2)', '(4, 3)', '(4, 4)', '(4, 5)', '(4, 6)'),
                          select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(5, 7), no_scrollbar=True)]]
col_names5 = [[sg.Listbox(values=('(5, 1)', '(5, 2)', '(5, 3)'),
                          select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(5, 4), no_scrollbar=True)]]
"""
# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Input(size=(50, 2), key="-FILE-"),
     sg.FileBrowse(file_types=file_types)],
    # [sg.Button("Обработать изображение")],
    [sg.Button("Загрузить изображение")],
    [sg.Button("Найти разрешение")],
    # [sg.Button("Сохранить для датасета")],
    [sg.Text("Выберите порог")],
    # [sg.Slider((1, 40), 128, 1, orientation='h', size=(30, 15), key='thresh_slider')],
    [sg.InputText(size=(15, 1), key='thresh_slider')],
    #[sg.Text("Номер группы наибольшего вертикального разрешения")],
    #[sg.Text(size=(20, 1), key="-TOUT0-"), sg.Text(size=(20, 1), key="-TOUT01-")],
    #[sg.Text("Номер группы наибольшего горизонтального разрешения")],
    #[sg.Text(size=(20, 1), key="-TOUT1-"), sg.Text(size=(20, 1), key="-TOUT11-")],
    [sg.Text("Разрешение:")],
    [sg.Text(size=(20, 1), key="-TOUT2-")],
    [sg.Text(size=(40, 2), key="-TOUT3-")],
    # [sg.Column(col_names0, scrollable=False),
    # sg.Multiline(size=(10, 7), key='-IN00-', write_only=True, no_scrollbar=True),
    # sg.Multiline(size=(10, 7), key='-IN01-', write_only=True, no_scrollbar=True),
    # sg.Column(col_names1, scrollable=False),
    # sg.Multiline(size=(10, 7), key='-IN10-', write_only=True, no_scrollbar=True),
    # sg.Multiline(size=(10, 7), key='-IN11-', write_only=True, no_scrollbar=True)],
    # [sg.Column(col_names2, scrollable=False),
    # sg.Multiline(size=(10, 7), key='-IN20-', write_only=True, no_scrollbar=True),
    # sg.Multiline(size=(10, 7), key='-IN21-', write_only=True, no_scrollbar=True),
    # sg.Column(col_names3, scrollable=False),
    # sg.Multiline(size=(10, 7), key='-IN30-', write_only=True, no_scrollbar=True),
    # sg.Multiline(size=(10, 7), key='-IN31-', write_only=True, no_scrollbar=True)]

    # [sg.Column(col_names4, scrollable=False),
    # sg.Multiline(size=(10, 7), key='-IN40-', write_only=True, no_scrollbar=True),
    # sg.Multiline(size=(10, 7), key='-IN41-', write_only=True, no_scrollbar=True),
    # sg.Column(col_names5, scrollable=False),
    # sg.Multiline(size=(10, 4), key='-IN50-', write_only=True, no_scrollbar=True),
    # sg.Multiline(size=(10, 4), key='-IN51-', write_only=True, no_scrollbar=True)]
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
    #print(window_h)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder

    if event == "Загрузить изображение":
        window_w, window_h = window.size
        # window['-IN20-'].update('')
        # window['-IN21-'].update('')
        # window['-IN30-'].update('')
        # window['-IN31-'].update('')
        # window['-IN40-'].update('')
        # window['-IN41-'].update('')
        # window['-IN50-'].update('')
        # window['-IN51-'].update('')
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

            # thresholds = res.find_resolution_old(threshold)
            threshold_h, threshold_v = res.find_resolution_network(threshold)
            # image = resize_img(res.image)
            image = resize_img(res.image, window_h)
            imgbytes = cv.imencode('.png', image)[1].tobytes()
            window["-IMAGE-"].update(data=imgbytes)

            #window["-TOUT0-"].update(res.vertical_resolution)
            #window["-TOUT01-"].update(res.get_table_resolution(res.vertical_resolution))
            #window["-TOUT1-"].update(res.horizontal_resolution)
            #window["-TOUT11-"].update(res.get_table_resolution(res.horizontal_resolution))
            window["-TOUT2-"].update(
                '%5.1f' % ((res.get_table_resolution(res.vertical_resolution) + res.get_table_resolution(
                    res.horizontal_resolution)) / 2))
            """
            for i in range(0, 6):
                if thresholds[i][0] == None:
                    window['-IN00-'].print('%5.2f' % -1)
                else:
                    window['-IN00-'].print('%5.2f' % thresholds[i][0])
                if thresholds[i][1] == None:
                    window['-IN01-'].print('%5.2f' % -1)
                else:
                    window['-IN01-'].print('%5.2f' % thresholds[i][1])
            for i in range(6, 12):
                if thresholds[i][0] == None:
                    window['-IN10-'].print('%5.2f' % -1)
                else:
                    window['-IN10-'].print('%5.2f' % thresholds[i][0])
                if thresholds[i][1] == None:
                    window['-IN11-'].print('%5.2f' % -1)
                else:
                    window['-IN11-'].print('%5.2f' % thresholds[i][1])
                    

            for i in range(0, 6, 1):
                if i >= len(threshold_v):
                    break
                print(i)
                if threshold_v[i] == None:
                    window['-IN20-'].print('%5.2f' % -1)
                else:
                    window['-IN20-'].print('%5.2f' % threshold_v[i])
                if threshold_h[i] == None:
                    window['-IN21-'].print('%5.2f' % -1)
                else:
                    window['-IN21-'].print('%5.2f' % threshold_h[i])
            for i in range(6, 12, 1):
                print(i)
                if i >= len(threshold_v):
                    break
                if threshold_v[i] == None:
                    window['-IN30-'].print('%5.2f' % -1)
                else:
                    window['-IN30-'].print('%5.2f' % threshold_v[i])
                if threshold_h[i] == None:
                    window['-IN31-'].print('%5.2f' % -1)
                else:
                    window['-IN31-'].print('%5.2f' % threshold_h[i])
            
            for i in range(24, 36, 2):
                print(i)
                if i > len(threshold_v):
                    break
                if threshold_v[i] == None:
                    window['-IN40-'].print('%5.2f' % -1)
                else:
                    window['-IN40-'].print('%5.2f' % threshold_v[i])
                if threshold_h[i] == None:
                    window['-IN41-'].print('%5.2f' % -1)
                else:
                    window['-IN41-'].print('%5.2f' % threshold_h[i])
            for i in range(36, 42, 2):
                print(i)
                if i > len(threshold_v):
                    break
                if threshold_v[i] == None:
                    window['-IN50-'].print('%5.2f' % -1)
                else:
                    window['-IN50-'].print('%5.2f' % threshold_v[i])
                if threshold_h[i] == None:
                    window['-IN51-'].print('%5.2f' % -1)
                else:
                    window['-IN51-'].print('%5.2f' % threshold_h[i])
            """

window.close()
