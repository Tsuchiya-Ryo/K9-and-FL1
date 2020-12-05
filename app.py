import os, base64, shutil, gc
import glob
import io
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from base64 import decodebytes
import numpy as np
from efficientnet_pytorch import EfficientNet

# pip install efficientnet_pytorch

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# INTERFACE ---------------------------------------
app.layout = dbc.Container(children=[
    html.Br(),
    html.H2(children="Dog/Cat classification"),
    html.Br(),
    html.Div(children="月並みですが。"),
    html.Div(children="連続投入時は更新挟んだほうがいいです。"),
    html.Div([html.A("source", href="https://github.com/Tsuchiya-Ryo/K9-and-FL1")]),
    html.Br(),
    
    dcc.Upload(
        id="upload-data",
        children=html.Div(
            html.A("Upload jpg/jpeg/png file")
        ),
        style = {
                    'width': '60%',
                    'height': '40px',
                    'lineHeight': '30px',
                    'borderWidth': '1px',
                    'borderStyle': 'solid',
                    'borderRadius': '9px',
                    'textAlign': 'center',
                    'margin': '30px',
                    "backgroundcolor":"blue"
            },
        multiple=True,
        max_size=100000000
    ),
    
    html.Br(),
    html.Div([dcc.Graph(id="example-graph")]),
    html.Div(id="selected-img"),
])


@app.callback(
    Output("example-graph", "figure"),
    Input("upload-data","contents"),
    Input("upload-data","filename")
)
def update_graph(images, filename):

    if os.path.exists("./assets"):
        shutil.rmtree("./assets")
    if not os.path.isdir("./assets"):
        os.mkdir("./assets")

    for name, image_str in zip(filename, images):
        image = image_str.split(",")[1]
        data = decodebytes(image.encode("ascii"))
        # with open("./assets/{}.png".format(name.split(".")[0]), "wb") as f:
        with open("./assets/{}.png".format(name), "wb") as f:
            f.write(data)
        del image
        del data

    img_target_path = "./assets/*.png"
    img_url_list = []
    for path in glob.glob(img_target_path):
        img_url_list.append(path)
    img_list =[]

    for url in img_url_list:
        img = Image.open(url)
        if len(np.array(img).shape) != 3:
            img = transforms.Grayscale(num_output_channels=3)(img)

        if np.array(img).shape[2] != 3:
            img = img.convert("RGB")

        img = tfms(img).unsqueeze(0)
        img_list.append(img)
        del img
    img_batch = torch.cat(img_list, axis=0)

    del img_list

    net.eval()
    with torch.no_grad():
        outputs = net(img_batch)
    datapoints = np.array(torch.abs(nn.Sigmoid()(outputs)-1))

    del img_batch
    del outputs

    gc.collect()

    return {"data":[go.Scatter(
        x = datapoints[:,0],
        y = datapoints[:,1],
        text = [fname[9:][:-4] for fname in img_url_list],
        mode = "markers",
        marker={
            "size":12,
            "opacity":0.5
        }
    )],
    "layout":go.Layout(
        xaxis={"title": "Cat (0) <------------------> Dog (1)",
                "range":[0,1],},
        yaxis={"title": "Young (0) <------------------> Adult (1)",
                "range":[0,1],},
    ),
    }

@app.callback(
    Output("selected-img", 'children'),
    [Input('example-graph', 'hoverData'),
    ]
)
def update_img(hoverData):
    try:
        imgurl = hoverData["points"][0]["text"] + ".png"
        return html.Img(src= app.get_asset_url(imgurl))
    except:
        return "hover_some_plots"

if __name__ == "__main__":
    net = EfficientNet.from_pretrained("efficientnet-b0")
    net._fc = nn.Linear(1280, 2)
    net.load_state_dict(torch.load(
        "efficient1204.pth", map_location = torch.device("cpu")))
    net.eval()

    tfms = transforms.Compose([transforms.Resize([224,224]), transforms.CenterCrop([224,224]),
            transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    app.run_server(host="0.0.0.0")