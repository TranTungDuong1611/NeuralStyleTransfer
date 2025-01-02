# Neural Style Transfer

## Overview

This project applies neural style transfer (NST) to transform an input image by adopting the artistic style of another image. NST leverages convolutional neural networks (CNNs) to separate and recombine content and style, producing visually striking results.

<tr>
<table align="center">
  <tr>
    <td align="center"><img src="../NeuralStyleTransfer/Image/content_image.jpg" width="100%"></td>
    <td align="center" style="font-size: 30px; font-weight: bold;"> + </td>
    <td align="center"><img src="../NeuralStyleTransfer/Image/style_image.jpg" width="140%"></td>
    <td align="center" style="font-size: 30px; font-weight: bold;"> = </td>
    <td colspan="4" align="center"><img src="../NeuralStyleTransfer/Image/generated_image.jpg" width="100%"></td>
  </tr>
</table>

## Streamlit App

`streamlit run streamlit.py`