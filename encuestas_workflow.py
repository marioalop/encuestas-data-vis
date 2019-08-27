import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


np.random.seed(14)
DF_FILE = 'TP_Intro_Aprendizaje_Supervisado.csv'
lb = LabelBinarizer()


def map_list(col):
    col = col.unique().tolist()
    col.sort()
    return {i:col.index(i) for i in col}


def normalize_data(df):
    df = df.drop('index', axis=1).drop('fecha_alerta', axis=1) \
        .drop('fecha_vencimiento', axis=1).drop('mes', axis=1) \
        .drop('vendedor_rc', axis=1).drop('responsable_de_gestionar_alerta', axis=1) \
        .drop('cliente', axis=1).drop('vin', axis=1) \
        .drop('vehículo', axis=1).drop('causas', axis=1) \
        .drop('tiempo_de_respuesta_dias', axis=1)

    df.marca = df.marca.str.replace("Usados", "7").str.replace("Marca", "").str.strip().astype(int)
    df.status = df.status.replace({'GESTIONADO': 1, 'SIN GESTION': 0})
    df.resp_alerta = df.resp_alerta.str.replace('Responsable', "").str.strip().astype(int)
    df.plazo_de_gestion = df.plazo_de_gestion.str.lower().replace(
        {'a tiempo': 1, 'sin tratamiento': 2, 'fuera de plazo': 3})
    df.sucursal = df.sucursal.str.replace('Sucursal', "").str.strip().astype(int)
    df.canal = df.canal.replace({'PV': 1, 'PA': 2, 'VN': 3, 'Usados': 4})
    df.incidente_o_retorno_inicial = df.incidente_o_retorno_inicial.replace({'SI': 1, 'NO': 0})
    df.tipo_de_alerta = df.tipo_de_alerta.replace({'OBSERVACION': 1,
                                                   'BAJA CALIFICACION': 2,
                                                   'BAJA CALIFICACION Y RETORNO': 3,
                                                   'BAJA CALIFICACION E INCIDENTE': 4,
                                                   'INCIDENTE': 5})

    df.categoria = df.categoria.str.replace("\\n", "").str.replace("\\r", "") \
        .str.lower().replace(map_list(df.categoria.str.replace("\\n", "") \
                                      .str.replace("\\r", "").str.lower()))

    df.puesto_involucrado = df.puesto_involucrado.str.lower().replace(map_list(df.puesto_involucrado.str.lower()))
    df.estado = df.estado.str.lower().replace(map_list(df.estado.str.lower()))

    return df


def save_model(model, name):
    filename = '{}'.format(name)
    pickle.dump(model, open(filename, 'wb'))
    return True


def load_model(filename):
    try:
        return pickle.load(open(filename, 'rb'))
    except Exception as e:
        print(e)
        return None


def preproccesing_df(df_file):
    encuestas_df = pd.read_csv(df_file)
    # pd.set_option('display.max_columns', 30)
    df = normalize_data(encuestas_df)
    for col in df.columns:
        print("{}: {}".format(col, df[col].unique()))
    df.to_csv("procesado.csv")
    n = len(df)
    df_train = df.iloc[:n]
    df_test = df.iloc[n:]

    y = df_train['tipo_de_alerta']
    X = df_train.drop('tipo_de_alerta', axis=1)
    yy = None
    XX = df_test.drop('tipo_de_alerta', axis=1)
    return y, X, yy, XX


def get_rf_classifier(X_train, y_train):
    model_path = "models/rf.model"
    model = load_model(model_path)

    if not model:
        tree_param = {'min_samples_leaf': range(5, 10),
                      "class_weight": [None, "balanced", "balanced_subsample"],
                      "warm_start": (False, True),
                      "oob_score": (False, True)
                      }

        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0, min_samples_split=100,
                                     criterion="gini", n_jobs=-1)
        tree_clf = GridSearchCV(clf, tree_param, scoring='accuracy', cv=3, iid=False, n_jobs=-1)

        tree_clf.fit(X_train, y_train)
        model = tree_clf.best_estimator_
        save_model(model, model_path)

    return model


def get_histgb_classifier(X_train, y_train):
    model_path = "models/histgb.model"
    model = load_model(model_path)
    if not model:
        #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
        params = dict(max_iter=[100, 500, 1000], max_leaf_nodes=[35, 45, 55, 65, 75], )
        clf = HistGradientBoostingClassifier(min_samples_leaf=30)
        tree_clf = GridSearchCV(clf, params, scoring='accuracy', cv=3, iid=False, n_jobs=-1)
        tree_clf.fit(X_train, y_train)
        model = tree_clf.best_estimator_
        save_model(model, model_path)

    return model


def get_svc_classifier(X_train, y_train):
    model_path = "models/linearsvc.model"
    model = load_model(model_path)
    #if not model:
    if True:
        #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
        param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10], 'loss': ['hinge', 'squared_hinge']}]
        linear_svc = LinearSVC(max_iter=10000)
        grid_search = GridSearchCV(linear_svc, param_grid, scoring='accuracy', cv=3, iid=False, n_jobs=4, verbose=5)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        save_model(model, model_path)

    return model


def get_ovsr_classifier(X, y):
    model_path = "models/ovsr.model"
    model = load_model(model_path)
    if not model:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
        model = OneVsRestClassifier(LinearSVC(C=10, loss='hinge', max_iter=10000))
        model.fit(X_train, y_train)

        save_model(model, model_path)
    return model


MODELS = {"OneVsRest": get_ovsr_classifier,
          "SVC": get_svc_classifier,
          "HistGradientBoosting": get_histgb_classifier,
          "RandomForest": get_rf_classifier
          }

def load_data(model_name):
    y, X, yy, XX = preproccesing_df(DF_FILE)
    #X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
    #model = MODELS[model_name](X_train, y_train)
    return {
            "X": X,
            "y": y,
            "yy": yy,
            "XX": XX
    }


def split_data(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_valid, y_train, y_valid
