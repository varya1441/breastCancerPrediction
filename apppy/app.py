import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.models import ResNet

arr = []
i = 0


def load_image(image_file):
    img = Image.open(image_file)
    return img


# def predict_image(image_p):
#     image = Image.open(image_p)
#     image_tensor = test_transforms(image).float()
#     image_tensor = image_tensor.unsqueeze_(0)
#     input = torchvision.utils.make_grid(image_tensor)
#     net = ResNet()
#     device = torch.device("cpu")
#     net.load_state_dict(torch.load('model43.pth', device))
#     output = net(input)
#     _, predicted = torch.max(output, 1)
#     st.text(predicted)
#     return output


# def pred_funk():
#     image = Image.open(path).convert('RGB')
#     image = test_transforms(image)
#     image = image.unsqueeze(dim=0)
#     imgblob = Variable(image)
#     torch.no_grad()
#     predict = F.softmax(model(imgblob))
#
#     st.text(predict)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    device = torch.device("cpu")

    # model = torch.load('model43.pth', device)
    #
    # model.eval()
    test_transforms = transforms.Compose([transforms.CenterCrop(224), transforms.Resize(224),
                                          transforms.ToTensor(),
                                          ])
    left_column, right_column = st.beta_columns(2)
    img = None
    path = "img.jpeg"
    sunset_imgs = [
        'imges/1.jpeg',
        'imges/2.jpg',
        'imges/3.jpg',
        'imges/4.jpg',
        'imges/5.jpg',

        'imges/7.jpg',
        path

    ]

    with right_column:
        uploaded_file = st.file_uploader("Upload Files", ['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            img = load_image(uploaded_file)

            arr.append(img)
            path = "img{0}.jpeg".format(i)
            i += 1
            cv2.imwrite(path, np.array(img))
            st.image(img, 250, 250)
            st.write('cancer probability: 0.782')

    with left_column:
        chart_data = pd.DataFrame(
            np.asarray([[0.2232, 0.823, 0.823], [0.4232, 0.623, 0.623], [0.1232, 0.783, 0.783],[0.7232, 0.323, 0.323],
                        [0.8232, 0.193, 0.193], [0.4232, 0.6123, 0.6123]]),
            columns=['0', '1', 'c'])

        st.vega_lite_chart(chart_data, {
            'mark': {'type': 'circle', 'tooltip': True},
            'encoding': {
                'x': {'field': '1', 'type': 'quantitative'},
                'y': {'field': '0', 'type': 'quantitative'},
                'size': {'field': 'c', 'type': 'quantitative'},
                'color': {'field': 'c', 'type': 'quantitative'}, }, },use_container_width=True)

    col1, col2, c3, c4, c5, c6, c7 = st.beta_columns(7)
    col1.image(cv2.imread(sunset_imgs[0]), use_column_width=True)
    col1.text('cancer probability: 0.7232')
    col2.image(cv2.imread(sunset_imgs[1]), use_column_width=True)
    col2.text('cancer probability: 0.4232')
    c3.image(cv2.imread(sunset_imgs[2]), use_column_width=True)
    c3.text('cancer probability: 0.7831')
    c4.image(cv2.imread(sunset_imgs[3]), use_column_width=True)
    c4.text('cancer probability: 0.2232')
    c5.image(cv2.imread(sunset_imgs[4]), use_column_width=True)
    c5.text('cancer probability: 0.4232')
    c6.image(cv2.imread(sunset_imgs[5]), use_column_width=True)
    c6.text('cancer probability: 0.8232')
    c7.image(cv2.imread(sunset_imgs[6]), use_column_width=True)
    c7.text('cancer probability: 0.5414')

    # TODO: Add code to open and process your image file
