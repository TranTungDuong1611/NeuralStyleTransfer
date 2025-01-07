# Neural Style Transfer

## Overview

This project applies neural style transfer (NST) to transform an input image by adopting the artistic style of another image. NST leverages convolutional neural networks (CNNs) to separate and recombine content and style, producing visually striking results.

## Results

The generated image is saved in the folder `results`

<tr>
<table align="center">
  <tr>
    <td align="center"><img src="Image/content_image.jpg" width="75%"></td>
    <td align="center" style="font-size: 30px; font-weight: bold;"> + </td>
    <td align="center"><img src="Image/style_image.jpg" width="120%"></td>
    <td align="center" style="font-size: 30px; font-weight: bold;"> = </td>
    <td colspan="4" align="center"><img src="results/generated_image.jpg" width="75%"></td>
  </tr>
</table>

<tr>
<table align="center">
  <tr>
    <td align="center"><img src="Image/content_image.jpg" width=85%"></td>
    <td align="center" style="font-size: 30px; font-weight: bold;"> + </td>
    <td align="center"><img src="Image/style2_image.webp" width="70%"></td>
    <td align="center" style="font-size: 30px; font-weight: bold;"> = </td>
    <td colspan="4" align="center"><img src="results/generated_image2.png" width="85%"></td>
  </tr>
</table>

## Streamlit App

Deploy model on streamlit

`streamlit run streamlit.py`

The interface of the streamlit app looks like this, you can upload an image for `content` and another image for `style` and choose the `step size` before start transfer:

![Streamlit](Image\streamlit.png)

## Huggingface Space

Deploy model on huggingface, you can visit via link:

`https://huggingface.co/spaces/TungDuong/image_style_transfer`

The interface in the huggingface look like this:

![Huggingface](Image\huggingface.png)
