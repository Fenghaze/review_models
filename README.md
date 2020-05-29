# review_models
复现语义分割的相关网络模型：

- fcn_vggnet：vgg16（backbone，**去掉FC层**）+ pool的**"skip"连接** + **upsampling**
- segnet：encoder（vgg16的前13层，改造最大池化层，返回池化索引），decoder（利用encoder的池化索引进行上采样）