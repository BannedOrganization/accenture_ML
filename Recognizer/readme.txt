Прототип распознавателя гос. номеров автотранспорта Российской федерации.
Запускается через файл ALPNR 2.0.py . В нём нужно передать путь к фотографии или видео в качестве аргумента функции Recognizer.
Функция возвращает json с гос. номером в случае его нахождения.

Под капотом 2 свёрточных нейронных сети написанных на Keras. Одна для распознавания цифр, вторая для распознавания букв номера.

1.jpg  фотография для примера, путь к которой можно передать функции Recognizer в качестве аргумента.
haarcascade_russian_plate_number.xml  каскадный классификатор Хаара, который отвечает за поиск области с гос. номером.

https://drive.google.com/file/d/15toNp1TgEEId_2CYliXunkpZtSD0dnGj/view?usp=sharing        ссылка на нейронную сеть digit_.h5
https://drive.google.com/file/d/1xswSlN2uCGfhaFkmv4NF6OuRHigy15jl/view?usp=sharing        ссылка на нейронную сеть letter_.h5

https://youtu.be/P9kHmJ42Sqg              Демонстрация работы прототипа распознавателя гос. номеров.
