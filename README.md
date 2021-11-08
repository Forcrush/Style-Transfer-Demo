<!--
 * @Author: Puffrora
 * @Date: 2021-11-06 14:00:38
 * @LastModifiedBy: Puffrora
 * @LastEditTime: 2021-11-08 13:00:11
-->
# A demo of Style Transfer Model based on pre-trained Vgg19

## Structure

`main.py` Entrance

`train.py` Training process and loss function

`model.py` Model construction

`utils.py` Some helper functions

`settings.py` Some dafault parameters used in model construction or preprocessing

## Execution

```
python3 main.py
```
`-c 1` # category of content image, it should match your content image name, e.g. if your content image is `images/content/content26.jpg`, you should set `-c 26`. Default is 1.

`-s 1` # category of style image, usage is simialr with `-c`. Default is 1.

`-epoch 200` # traning epoches. Default is 200.

`-iter 100` # traning iterations per epoch. Default is 100.

`-lr 0.01` # learning rate. Default is 0.01.

## Reference
1. [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v1.pdf)
2. https://github.com/AaronJny/nerual_style_change