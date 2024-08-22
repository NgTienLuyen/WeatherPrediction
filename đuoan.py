import tkinter as tk
from tkinter import Label, Entry, LabelFrame, ttk
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

# Tải các mô hình đã lưu
svm_model = joblib.load('G:/NGHIÊN CỨU KHOA HỌC/NCKH/MODEL/svm_model.pkl')
cart_model = joblib.load('G:/NGHIÊN CỨU KHOA HỌC/NCKH/MODEL/cart_model.pkl')
id3_model = joblib.load('G:/NGHIÊN CỨU KHOA HỌC/NCKH/MODEL/id3_model.pkl')
pca_best_svm = joblib.load('G:/NGHIÊN CỨU KHOA HỌC/NCKH/MODEL/pca_best_svm.pkl')
pca_best_cart = joblib.load('G:/NGHIÊN CỨU KHOA HỌC/NCKH/MODEL/pca_best_cart.pkl')
pca_best_id3 = joblib.load('G:/NGHIÊN CỨU KHOA HỌC/NCKH/MODEL/pca_best_id3.pkl')

# Tạo form
form = tk.Tk()
form.title("Dự đoán thời tiết:")
form.geometry("1700x900")

lable_people = LabelFrame(form, text="Nhập thông tin dự báo thời tiết", font=("Arial Bold", 13), fg="red")
lable_people.pack(fill="both", expand="yes")
lable_people.config(bg="#FEF2D1")

# THÔNG TIN CỘT 1
lable_Temp9am = Label(form, font=("Arial Bold", 11), text="Nhiệt độ lúc 9h(°C):", bg="#FEF2D1")
lable_Temp9am.place(x=50, y=50)
textbox_Temp9am = Entry(form, width=30, font=("Arial Bold", 11))
textbox_Temp9am.place(x=430, y=50)

lable_Temp3pm = Label(form, font=("Arial Bold", 11), text="Nhiệt độ lúc 15h(°C):", bg="#FEF2D1")
lable_Temp3pm.place(x=50, y=90)
textbox_Temp3pm = Entry(form, width=30, font=("Arial Bold", 11))
textbox_Temp3pm.place(x=430, y=90)

lable_MinTemp = Label(form, font=("Arial Bold", 11), text="Nhiệt độ thấp nhất trong ngày(°C):", bg="#FEF2D1")
lable_MinTemp.place(x=50, y=130)
textbox_MinTemp = Entry(form, width=30, font=("Arial Bold", 11))
textbox_MinTemp.place(x=430, y=130)

lable_MaxTemp = Label(form, font=("Arial Bold", 11), text="Nhiệt độ cao nhất trong ngày(°C):", bg="#FEF2D1")
lable_MaxTemp.place(x=50, y=170)
textbox_MaxTemp = Entry(form, width=30, font=("Arial Bold", 11))
textbox_MaxTemp.place(x=430, y=170)

lable_Rainfall = Label(form, font=("Arial Bold", 11), text="Lượng mưa trong ngày(mm):", bg="#FEF2D1")
lable_Rainfall.place(x=50, y=210)
textbox_Rainfall = Entry(form, width=30, font=("Arial Bold", 11))
textbox_Rainfall.place(x=430, y=210)

lable_Evaporation = Label(form, font=("Arial Bold", 11), text="Độ bốc hơi trong ngày(mm):", bg="#FEF2D1")
lable_Evaporation.place(x=50, y=250)
textbox_Evaporation = Entry(form, width=30, font=("Arial Bold", 11))
textbox_Evaporation.place(x=430, y=250)

lable_Sunshine = Label(form, font=("Arial Bold", 11), text="Số giờ có nắng:", bg="#FEF2D1")
lable_Sunshine.place(x=50, y=290)
textbox_Sunshine = Entry(form, width=30, font=("Arial Bold", 11))
textbox_Sunshine.place(x=430, y=290)

lable_WindGustDir = Label(form, font=("Arial Bold", 11), text="Hướng gió mạnh nhất trong ngày:", bg="#FEF2D1")
lable_WindGustDir.place(x=50, y=330)
option_WindGustDir = ['W', 'NNW', 'WNW', 'ENE', 'NNE', 'NW', 'SSE', 'NE', 'ESE', 'WSW', 'SE', 'SW', 'N', 'E', 'SSW',
                      'S']
combobox_WindGustDir = ttk.Combobox(form, width=28, font=("Arial Bold", 11), values=option_WindGustDir,
                                    state="readonly")
combobox_WindGustDir.place(x=430, y=330)

lable_WindDir9am = Label(form, font=("Arial Bold", 11), text="Hướng gió trong 10 phút trước 9h:", bg="#FEF2D1")
lable_WindDir9am.place(x=50, y=370)
option_WindDir9am = ['W', 'NNW', 'WNW', 'ENE', 'NNE', 'NW', 'SSE', 'NE', 'ESE', 'WSW', 'SE', 'SW', 'N', 'E', 'SSW', 'S']
combobox_WindDir9am = ttk.Combobox(form, width=28, font=("Arial Bold", 11), values=option_WindDir9am, state="readonly")
combobox_WindDir9am.place(x=430, y=370)

lable_WindDir3pm = Label(form, font=("Arial Bold", 11), text="Hướng gió trong 10 phút trước 15h:", bg="#FEF2D1")
lable_WindDir3pm.place(x=50, y=410)
option_WindDir3pm = ['W', 'NNW', 'WNW', 'ENE', 'NNE', 'NW', 'SSE', 'NE', 'ESE', 'WSW', 'SE', 'SW', 'N', 'E', 'SSW', 'S']
combobox_WindDir3pm = ttk.Combobox(form, width=28, font=("Arial Bold", 11), values=option_WindDir3pm, state="readonly")
combobox_WindDir3pm.place(x=430, y=410)

# THÔNG TIN CỘT 2
lable_WindGustSpeed = Label(form, font=("Arial Bold", 11), text="Tốc độ gió mạnh nhất trong ngày:", bg="#FEF2D1")
lable_WindGustSpeed.place(x=700, y=50)
textbox_WindGustSpeed = Entry(form, width=30, font=("Arial Bold", 11))
textbox_WindGustSpeed.place(x=1020, y=50)

lable_WindSpeed9am = Label(form, font=("Arial Bold", 11), text="Tốc độ gió trong 10 phút trước 9h(Km/h):", bg="#FEF2D1")
lable_WindSpeed9am.place(x=700, y=90)
textbox_WindSpeed9am = Entry(form, width=30, font=("Arial Bold", 11))
textbox_WindSpeed9am.place(x=1020, y=90)

lable_WindSpeed3pm = Label(form, font=("Arial Bold", 11), text="Tốc độ gió trong 10 phút trước 15h(Km/h):",
                           bg="#FEF2D1")
lable_WindSpeed3pm.place(x=700, y=130)
textbox_WindSpeed3pm = Entry(form, width=30, font=("Arial Bold", 11))
textbox_WindSpeed3pm.place(x=1020, y=130)

lable_Humidity9am = Label(form, font=("Arial Bold", 11), text="Độ ẩm của gió lúc 9h(%):", bg="#FEF2D1")
lable_Humidity9am.place(x=700, y=170)
textbox_Humidity9am = Entry(form, width=30, font=("Arial Bold", 11))
textbox_Humidity9am.place(x=1020, y=170)

lable_Humidity3pm = Label(form, font=("Arial Bold", 11), text="Độ ẩm của gió lúc 15h(%):", bg="#FEF2D1")
lable_Humidity3pm.place(x=700, y=210)
textbox_Humidity3pm = Entry(form, width=30, font=("Arial Bold", 11))
textbox_Humidity3pm.place(x=1020, y=210)

lable_Pressure9am = Label(form, font=("Arial Bold", 11), text="Áp suất khí quyển lúc 9h(hPa):", bg="#FEF2D1")
lable_Pressure9am.place(x=700, y=250)
textbox_Pressure9am = Entry(form, width=30, font=("Arial Bold", 11))
textbox_Pressure9am.place(x=1020, y=250)

lable_Pressure3pm = Label(form, font=("Arial Bold", 11), text="Áp suất khí quyển lúc 15h(hPa):", bg="#FEF2D1")
lable_Pressure3pm.place(x=700, y=290)
textbox_Pressure3pm = Entry(form, width=30, font=("Arial Bold", 11))
textbox_Pressure3pm.place(x=1020, y=290)

lable_Cloud9am = Label(form, font=("Arial Bold", 11), text="Tỉ lệ mây lúc 9h(%):", bg="#FEF2D1")
lable_Cloud9am.place(x=700, y=330)
textbox_Cloud9am = Entry(form, width=30, font=("Arial Bold", 11))
textbox_Cloud9am.place(x=1020, y=330)

lable_Cloud3pm = Label(form, font=("Arial Bold", 11), text="Tỉ lệ mây lúc 15h(%):", bg="#FEF2D1")
lable_Cloud3pm.place(x=700, y=370)
textbox_Cloud3pm = Entry(form, width=30, font=("Arial Bold", 11))
textbox_Cloud3pm.place(x=1020, y=370)

lable_Model = Label(form, font=("Arial Bold", 11), text="Chọn mô hình:", bg="#FEF2D1")
lable_Model.place(x=700, y=450)
option_Model = ['SVM', 'CART', 'ID3']
combobox_Model = ttk.Combobox(form, width=28, font=("Arial Bold", 11), values=option_Model, state="readonly")
combobox_Model.place(x=1020, y=450)

# Khởi tạo các bộ mã hóa
label_encoders = {
    'WindGustDir': LabelEncoder(),
    'WindDir9am': LabelEncoder(),
    'WindDir3pm': LabelEncoder(),
}

# Cập nhật bộ mã hóa với dữ liệu mẫu
def fit_label_encoders():
    for column, encoder in label_encoders.items():
        if column == 'WindGustDir':
            encoder.fit(['W', 'NNW', 'WNW', 'ENE', 'NNE', 'NW', 'SSE', 'NE', 'ESE', 'WSW', 'SE', 'SW', 'N', 'E', 'SSW', 'S'])
        elif column == 'WindDir9am':
            encoder.fit(['W', 'NNW', 'WNW', 'ENE', 'NNE', 'NW', 'SSE', 'NE', 'ESE', 'WSW', 'SE', 'SW', 'N', 'E', 'SSW', 'S'])
        elif column == 'WindDir3pm':
            encoder.fit(['W', 'NNW', 'WNW', 'ENE', 'NNE', 'NW', 'SSE', 'NE', 'ESE', 'WSW', 'SE', 'SW', 'N', 'E', 'SSW', 'S'])

# Gọi hàm này để khởi tạo bộ mã hóa trước khi sử dụng chúng để dự đoán
fit_label_encoders()


def predict():
    # Lấy giá trị từ các ô nhập liệu
    try:
        Temp9am = float(textbox_Temp9am.get())
        Temp3pm = float(textbox_Temp3pm.get())
        MinTemp = float(textbox_MinTemp.get())
        MaxTemp = float(textbox_MaxTemp.get())
        Rainfall = float(textbox_Rainfall.get())
        Evaporation = float(textbox_Evaporation.get())
        Sunshine = float(textbox_Sunshine.get())
        WindGustDir = combobox_WindGustDir.get()
        WindDir9am = combobox_WindDir9am.get()
        WindDir3pm = combobox_WindDir3pm.get()
        WindGustSpeed = float(textbox_WindGustSpeed.get())
        WindSpeed9am = float(textbox_WindSpeed9am.get())
        WindSpeed3pm = float(textbox_WindSpeed3pm.get())
        Humidity9am = float(textbox_Humidity9am.get())
        Humidity3pm = float(textbox_Humidity3pm.get())
        Pressure9am = float(textbox_Pressure9am.get())
        Pressure3pm = float(textbox_Pressure3pm.get())
        Cloud9am = float(textbox_Cloud9am.get())
        Cloud3pm = float(textbox_Cloud3pm.get())
        model_choice = combobox_Model.get()

        # Mã hóa các giá trị văn bản thành số nếu cần
        WindGustDir = label_encoders['WindGustDir'].transform([WindGustDir])[0]
        WindDir9am = label_encoders['WindDir9am'].transform([WindDir9am])[0]
        WindDir3pm = label_encoders['WindDir3pm'].transform([WindDir3pm])[0]

        # Xử lý dữ liệu đầu vào cho mô hình
        data = np.array([[Temp9am, Temp3pm, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
                          WindGustDir, WindDir9am, WindDir3pm, WindGustSpeed, WindSpeed9am,
                          WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm,
                          Cloud9am, Cloud3pm]])

        # Chọn mô hình
        if model_choice == 'SVM':
            pca = pca_best_svm
            model = svm_model
        elif model_choice == 'CART':
            pca = pca_best_cart
            model = cart_model
        elif model_choice == 'ID3':
            pca = pca_best_id3
            model = id3_model
        else:
            result_label.config(text="Chưa chọn mô hình!")
            return

        # Dự đoán
        data_pca = pca.transform(data)
        prediction = model.predict(data_pca)

        if prediction[0] == 0:
            # Hiển thị dự đoán không mưa
            result_label.config(text=f"Dự đoán: Ngày mai Không mưa")
        if prediction[0] == 1:
            # Hiển thị dự đoán có mưa
            result_label.config(text=f"Dự đoán: Ngày mai Có mưa")

    except ValueError as e:
        result_label.config(text=f"Lỗi đầu vào: {e}")
    except Exception as e:
        result_label.config(text=f"Đã xảy ra lỗi: {e}")

# Nút dự đoán
predict_button = tk.Button(form, text="Dự đoán", font=("Arial Bold", 11), command=predict)
predict_button.place(x=800, y=490)

# Nơi hiển thị kết quả dự đoán
result_label = Label(form, font=("Arial Bold", 11), bg="#FEF2D1")
result_label.place(x=800, y=530)


form.mainloop()
