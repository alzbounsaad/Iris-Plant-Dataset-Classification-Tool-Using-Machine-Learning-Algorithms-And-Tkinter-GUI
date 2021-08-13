import tkinter
from tkinter import filedialog
from tkinter import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error


def center(win):
    """
    centers a tkinter window
    :param win: the main window or Toplevel window to center
    """
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()


text = ".:: Guess Iris Plant Type ::."

tk = tkinter.Tk()
tk.geometry("900x400")
tk.title(text)
tk.resizable(0, 0)


# load CSV File data...
def load_dataFrame(filename):
    dataFrame = pd.read_csv(filename)

    return dataFrame


# Create a File Explorer label
label_file_explorer = Label(tk, borderwidth=1, relief="raised")


# file explorer window
def Load_CSV_Dataset():
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Load CSV Dataset",
                                          filetypes=(("CSV files",
                                                      "*.csv*"),
                                                     ("all files",
                                                      "*.*")))

    dataFrame = load_dataFrame(filename)

    # Change label contents
    label_file_explorer.configure(text=filename)


label_file_explorer.pack(padx=50, pady=116)


# calculate the machine learning algorithms accuracy...
def calculate_accuracy():
    dataFrame = pd.read_csv(label_file_explorer.cget("text"))

    X = dataFrame[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    Y = dataFrame.IrisClass

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # SVM Classifier
    linear_svc = SVC(kernel='linear').fit(x_train, y_train)
    svm_prediction = linear_svc.predict(x_test)

    # KNN, first try 5
    mod_5nn = KNeighborsClassifier(n_neighbors=5)
    mod_5nn.fit(x_train, y_train)
    knn_prediction = mod_5nn.predict(x_test)

    # RFC 
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
    rf.fit(x_train, y_train)
    rfc_prediction = rf.predict(x_test)

    R1 = Radiobutton(tk,
                     text='SVM Accuracy (% ' + "{:.2f}".format(metrics.accuracy_score(svm_prediction, y_test)) + ')',
                     value=0).place(x=75, y=200)
    R2 = Radiobutton(tk,
                     text='RFC Accuracy (% ' + "{:.2f}".format(metrics.accuracy_score(rfc_prediction, y_test)) + ')',
                     value=0).place(x=75, y=225)
    R3 = Radiobutton(tk,
                     text='KNN Accuracy (% ' + "{:.2f}".format(metrics.accuracy_score(knn_prediction, y_test)) + ')',
                     value=0).place(x=75, y=250)


# Define Model Function...
def define_model():
    dataFrame = pd.read_csv(label_file_explorer.cget("text"))

    X = dataFrame[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    Y = dataFrame.IrisClass

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    mod_dt = DecisionTreeClassifier(max_depth=50, random_state=1000)
    mod_dt.fit(x_train, y_train)
    prediction = mod_dt.predict(x_test)

    tkinter.messagebox.showinfo(title='Mean Absolute Error', message='Mean Absolute Error: ' + "{:.2f}".format(
        mean_absolute_error(prediction, y_test)))


frame = Frame(tk, borderwidth=5)
frame.pack(fill=BOTH, expand=1)
header = Label(tk, bg="#3776ab", fg='white', pady=5, font=('Helvetica', '18', 'bold'), text=text).place(x=300, y=30)


# Step 1 ...
step_1 = Label(tk, pady=5, font=('Helvetica', '10'), text='step 1: ').place(x=0, y=110)

# Browse button...
button_explore = Button(tk,
                        text="Load CSV Dataset",
                        command=Load_CSV_Dataset).place(x=700, y=110)

# Step 2 ...
step_2 = Label(tk, pady=5, font=('Helvetica', '10'),
               text='step 2: Apply machine learning algorithms & Select the best accuracy: ').place(x=0, y=150)

# Calculate accuracy button...
button_calculate_accuracy = Button(tk,
                                   text="Calculate Accuracy",
                                   command=calculate_accuracy).place(x=700, y=150)

# Step 3 ...
step_3 = Label(tk, pady=5, font=('Helvetica', '10'), text='step 3: define the model: ').place(x=0, y=300)

# Define accuracy button...
button_define_model = Button(tk,
                             text="Define Model",
                             command=define_model).place(x=700, y=300)

center(tk)

tk.mainloop()
