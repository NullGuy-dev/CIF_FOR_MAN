# {
#       "tag": "empty",
#       "patterns": [""]
#     },
# -*- coding: utf-8 -*-

# Підключення потрібних бібліотек/фреймворків

import nltk

nltk.download('punkt')
import tflearn, json, pickle
import numpy as np
from nltk.stem.lancaster import LancasterStemmer


# Створення класу нашого API

class CIF:
    def __init__(self):
        # Створюємо порожні змінні, для подальшого додавання даних

        self.data: dict = {}  # порожній словник який зберігатиме дані для навчання, при самому навчанні нейронної мережі

        # порожні базові змінні для навчання та використання
        self.training = []
        self.output = []
        self.words = []
        self.labels = []

        self.stemmer = LancasterStemmer()

        self.actv_func = "softmax"
        self.neurals = [8, 8]
        self.loss_func = "categorical_crossentropy"
        self.optmzr = "adam"


    def arr_with_word_ids(self, s): # метод який повертатиме матрицю зі значеннями схожості повідомлення та вибірки для навчання
        bag = [0 for _ in range(len(self.words))] # створюємо порожній список, який зберігатиме матрицю зі значеннями схожості повідомлення та вибірки для навчання
        s_words = nltk.word_tokenize(s) # змінна яка просто розділити слова через прогалини і зберігати до списку
        s_words = [word.lower() for word in s_words] # ставить слова у нижній рерістр та зберігає у змінну
        for se in s_words: # перебираем список
            for i, w in enumerate(self.words): # беремо слова з вибірки
                if w == se or w == se[:-1]: # порівнює
                    bag[i] = 1 # якщо слово знайдено, воно в матриці як 1  
        return np.array(bag) # конвертуємо його до списку numpy, щоб прискорити з ним роботу. Також повертаємо його


    def load_dnn_struct(self): # метод для створення структури нейронної мережі
        net = tflearn.input_data(shape=[None, len(self.training[0])]) # будуємо структуру входного слої
        for n in self.neurals: # робимо стільки слоїв нейронної мережі, скільки забажаємо
            net = tflearn.fully_connected(net, n) # їх додавання до нейронної мережі
        net = tflearn.fully_connected(net, len(self.output[0]), activation=self.actv_func) # будування вихідного слою, тобто слою, який буде видавати результат праці нейронної мережі
        net = tflearn.regression(net, loss=self.loss_func, optimizer=self.optmzr) # описуємо нашу мережу, тобто її функція втрат, оптимізатор тощо
        self.model = tflearn.DNN(net) # загрузка структури нейронної мережі, та її збереження

    def train_model(self, tflearn_file_name: str, pickle_file_name: str, data_for_training: dict, epochs=10000,
                    batch_size=8): # метод, який допомагає навчити нашу модель
        
        # просто записуємо шаблони наших змінних
        self.data = data_for_training
        self.training = []
        self.output = []
        docs_x = []
        docs_y = []

        # розподілити дані з навчальної вибірки до змінних
        for intent in self.data["data"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                self.words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])
            if intent["tag"] not in self.labels:
                self.labels.append(intent["tag"])
        
        self.words = [self.stemmer.stem(w.lower()) for w in self.words] # редагуємо слово, тобто видаляємо закінчення і т.п. та зберігаємо його до списку
        self.words = sorted(list(set(self.words))) # сортируем слова и удаляем повторения
        self.labels = sorted(self.labels) # сортируем список с словами для обучения

        out_empty = [0 for _ in range(len(self.labels))] # створюємо пусту матрицю

        for x, doc in enumerate(docs_x): # беремо дані та створюємо матрицю, на якій і буде навчатися нейронна мережа
            # просто будуємо матрицю для навчання
            bag = []
            wrds = [self.stemmer.stem(w.lower()) for w in doc]
            for w in self.words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
            
            self.training.append(bag) # заповнення матриці для навчання

            # виставлення марок(1), для кожного елемента частини навчальної виборки, щоб нейронній мережі було зрозуміло, до якого класу належить елемент
            output_row = out_empty[:]
            output_row[self.labels.index(docs_y[x])] = 1
            self.output.append(output_row) # залишаємо у матриці
            
        self.load_dnn_struct() # підгружаємо структуру нейронної мережі

        self.training = np.array(self.training) # робимо з нього масив numpy, щоб робота з масивом була швидша
        self.output = np.array(self.output) # робимо з нього масив numpy, щоб робота з масивом була швидша

        # зберегаємо у файл, щоб перед кожним використанням її не навчати
        with open(pickle_file_name, "wb") as f:
            pickle.dump((self.words, self.labels, self.training, self.output), f)

        self.model.fit(self.training, self.output, n_epoch=epochs, batch_size=batch_size) # починаємо процес навчання мережі
        self.model.save(tflearn_file_name) # зберігаємо модель

    # "model.tflearn"

    def run(self, message: str, symbols: str = ',-—=[]{}\;"\|/+*1234567890?`~!@#№$%^:&'):
        try:
            self.model
        except:
            raise ValueError("Model is not loaded. Please, download it and load it")

        if self.words == []:
            raise ValueError("Maybe, you didn't connected your pickle's data")

        for symbol in symbols:
            message = message.replace(symbol, " ").replace("'", "")
        results = self.model.predict([self.arr_with_word_ids(message)])
        results_index = np.argmax(results)
        tag = self.labels[results_index]
        res_per = format(float(results[0][results_index]), 'f')
        for tg in self.data["data"]:
            if tg['tag'] == tag:
                if float(res_per) >= 0.8:
                    return tag
                else:
                    return "empty"


testing = CIF()
with open("data.json", encoding="utf8") as f:
    dft = json.load(f)


testing.train_model("CIF_model.tflearn", "CIF_model.pickle", dft, epochs=5000, batch_size=4)

with open("CIF_model.pickle", "rb") as pkl_file:
    testing.words, testing.labels, testing.training, testing.output = pickle.load(pkl_file)

with open("testing.json", encoding="utf8") as ft:
    test_d = json.load(ft)

testing.load_dnn_struct()
testing.model.load("CIF_model.tflearn")
testing.data = dft

for mes_part in test_d["test_arr"]:
    all_m = []
    mes = mes_part[0].split(".")
    res = ""
    for part_of_mes in mes:
        if part_of_mes:
            part_of_mes = part_of_mes[:-1]
            res = testing.run(part_of_mes.lower())
            all_m.append(res)
    if "cursing" in all_m:
        res = "cursing"
    elif "insulting" in all_m:
        res = "insulting"
    else:
        res = "empty"
    print(f"{res} - {mes_part[1]}\n")
