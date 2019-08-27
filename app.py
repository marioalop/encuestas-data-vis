# PLTLY DASH CORE COMPONENTS
import time
import dash
import dash_core_components as dcc
import dash_html_components as html
from utils import dash_reusable_components as drc
from utils import figures as figs
#
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from encuestas_workflow import load_data, split_data, get_svc_classifier

cols = ['marca', 'status', 'plazo_de_gestion', 'sucursal', 'canal',
     'satisfaccion_inicial', 'recomendacion_inicial',
     'incidente_o_retorno_inicial', 'categoria', 'puesto_involucrado',
     'estado', 'resp_alerta']

cols_opts = [{"label": i.capitalize().replace("_", " "), "value": i} for i in cols]


app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.encuestas = load_data("SVC")
app.split_data = split_data
app.get_svc_classifier = get_svc_classifier
server = app.server


def generate_data(n_samples, dataset, noise):
    if dataset == "moons":
        return datasets.make_moons(n_samples=n_samples, noise=noise, random_state=0)

    elif dataset == "circles":
        return datasets.make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=1
        )

    elif dataset == "linear":
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1,
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            "Data type incorrectly specified. Please choose an existing dataset."
        )


app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Diplo encuestas - Support Vector Machine Explorer",
                                    href="https://github.com/plotly/dash-svm",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),

                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Select feature",
                                            id="dropdown-select-col1",
                                            options=cols_opts,
                                            clearable=False,
                                            searchable=False,
                                            value="canal",
                                        ),
                                        drc.NamedDropdown(
                                            name="Select feature",
                                            id="dropdown-select-col2",
                                            options=cols_opts,
                                            clearable=False,
                                            searchable=False,
                                            value="satisfaccion_inicial",
                                        )
                                    ],
                                ),
                                drc.Card(
                                    id="button-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Threshold",
                                            id="slider-threshold",
                                            min=0,
                                            max=1,
                                            value=0.5,
                                            step=0.01,
                                        ),
                                        html.Button(
                                            "Reset Threshold",
                                            id="button-zero-threshold",
                                        ),
                                    ],
                                ),

                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-svm",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
    ]
)






@app.callback(
    Output("slider-threshold", "value"),
    [Input("button-zero-threshold", "n_clicks")],
    [State("graph-sklearn-svm", "figure")],
)
def reset_threshold_center(n_clicks, figure):
    if n_clicks:
        Z = np.array(figure["data"][0]["z"])
        value = -Z.min() / (Z.max() - Z.min())
    else:
        value = 0.4959986285375595
    return value



@app.callback(
    Output("div-graphs", "children"),
    [
        Input("dropdown-select-col1", "value"),
        Input("dropdown-select-col2", "value"),
        Input("slider-threshold", "value"),

    ],
)
def update_svm_graph(col1, col2, threshold):
    t_start = time.time()
    h = 0.3  # step size in the mesh


    """"# Data Pre-processing
    X, y = generate_data(n_samples=sample_size, dataset=dataset, noise=noise)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42)
    print(X_train)
    print(X_train.shape)

    """


    print(app.encuestas["X"].columns)
    X = app.encuestas["X"][[col1, col2]].to_numpy()
    y = app.encuestas["y"].to_numpy()

    X_train, X_test, y_train, y_test = app.split_data(X, y)

    print(X.shape, y.shape)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    #"""

    np.savetxt("test.csv", X_train, delimiter=",")

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # xx = app.encuestas["XX"]
    # yy = app.encuestas["yy"]


    # Train SVM
    #clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, shrinking=flag)
    clf = app.get_svc_classifier(col1, col2, X_train, y_train)
    #..clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    prediction_figure = figs.serve_prediction_plot(
        model=clf,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        Z=Z,
        xx=xx,
        yy=yy,
        mesh_step=h,
        threshold=threshold,
    )

    #roc_figure = figs.serve_roc_curve(model=clf, X_test=X_test, y_test=y_test)

    #confusion_figure = figs.serve_pie_confusion_matrix(
    #    model=clf, X_test=X_test, y_test=y_test, Z=Z, threshold=threshold
    #)

    return [
        html.Div(
            id="svm-graph-container",
            children=dcc.Loading(
                className="graph-wrapper",
                children=dcc.Graph(id="graph-sklearn-svm", figure=prediction_figure),
                style={"display": "none"},
            ),
        ),
        #html.Div(
        #    id="graphs-container",
        #    children=[
        #        #dcc.Loading(
                #    className="graph-wrapper",
                #    children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
                #),
        #        dcc.Loading(
        #            className="graph-wrapper",
        ##            children=dcc.Graph(
        #                id="graph-pie-confusion-matrix", figure=confusion_figure
        #            ),
        #        ),
        #    ],
        #),
    ]


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
