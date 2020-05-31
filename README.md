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
- deelabv1：空洞卷积（扩大感受野），CRF（后处理，细化边界的语义标签）
    - 微调VGG16，使得最终输出为原图的1/8
        - 将fc6，fc7，fc8改为全卷积层，且fc8的输出通道为像素分类数
        - pool4，pool5的stride=2
        - 最后一个block的卷积层的dilation=2
        - fc6的output_channels=1024，kernelsize=3，dilation=12，padding=12