# review_models
复现语义分割的相关网络模型：

- fcn_vggnet：
    - vgg16（backbone，**去掉FC层**）+ 
    - pool的"skip"连接 
    - upsample到原图大小
- segnet：
    - encoder：vgg16的前13层，改造最大池化层，返回池化索引（return_indices=Treu）
    - decoder：利用encoder的池化索引进行上采样
- unet：用于生物医学成像
    - contraction path：5个blocks，每个block返回最后一层的feature map，并使用maxpool进行下采样
    - expansive path：4个blocks，使用Upsample或反卷积进行上采样，和contraction path的feature map连接后再进行卷积