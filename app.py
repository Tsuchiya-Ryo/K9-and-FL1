import os, base64, shutil
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

app = dash.Dash(__name__)
# INTERFACE ---------------------------------------
app.layout = dbc.Container(children=[
    html.Br(),
    html.H1(children="baby and adult K9/FL9"),
    html.Br(),
    
    dcc.Upload(
        id="upload-data",
        children=html.Div(
            html.A("Upload jpg/jpeg/png file")
        ),
        style = {
                    'width': '40%',
                    'height': '30px',
                    'lineHeight': '30px',
                    'borderWidth': '1px',
                    'borderStyle': 'solid',
                    'borderRadius': '9px',
                    'textAlign': 'center',
                    'margin': '30px',
                    "backgroundcolor":"blue"
            },
        multiple=True
    ),
    html.Div(id="selected-img"),
    html.Br(),
    html.Div([dcc.Graph(id="example-graph")]),
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

    # for i, image_str in enumerate(images):
    #     image = image_str.split(',')[1]
    #     # data = decodebytes(base64.b64decode(image))
    #     data = decodebytes(image.encode('ascii'))
    #     with open("./assets/image_{}.png".format(i+1), "wb") as f:
    #         f.write(data)

    for name, image_str in zip(filename, images):
        image = image_str.split(",")[1]
        data = decodebytes(image.encode("ascii"))
        with open("./assets/{}.png".format(name.split(".")[0]), "wb") as f:
            f.write(data)

    print(filename)
    img_target_path = "./assets/*.png"
    img_url_list = []
    for path in glob.glob(img_target_path):
        img_url_list.append(path)
    img_list =[]
    print(img_url_list)
    for url in img_url_list:
        img = Image.open(url)
        if len(np.array(img).shape) != 3:
            img = transforms.Grayscale(num_output_channels=3)(img)
        img = tfms(img).unsqueeze(0)
        img_list.append(img)
    img_batch = torch.cat(img_list, axis=0)
    print(img_batch.size())

    net.eval()
    with torch.no_grad():
        outputs = net(img_batch)
    datapoints = np.array(torch.abs(nn.Sigmoid()(outputs)-1))

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
        xaxis={"title": "Cat(0) <-------------------> Dog(1)",
                "range":[0,1]},
        yaxis={"title": "Young(0) <------------------> Adult(1)",
                "range":[0,1]}
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

    tfms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    app.run_server(host="0.0.0.0") 
