DPT:定义基础DPT模型结构及其参数。
- init：调用父类init函数(此处应该是torch自带module类的初始化函数)，通过_make_encoder函数生产网络结构参数。通过_make_fusion_block函数来生成scratch变量的refinenet参数。生成的head网络和键盘输入参数共同进行init。键盘输入参数主要包括features，backbone，readout，channels_last和use_bn
  - _make_encoder:主要是根据不同的backbone网络设置不同的pretrained参数以及scratch参数
    - _make_pretrained_vitl16_384:
      - _make_vit_b16_backbone:
    - _make_pretrained_vitb_rn50_384:
      - _make_vit_b_rn50_backbone:
    - _make_pretrained_vitb16_384:
      - _make_vit_b16_backbone:
    - _make_pretrained_resnext101_wsl：
      - _make_resnet_backbone：
    - _make_pretrained_efficientnet_lite3：
      - _make_efficientnet_backbone：
    - _make_scratch:以module为基础模型设置scratch的网络参数
  - _make_fusion_block：调用FeatureFusionBlock_custom函数实现fusion模块(主要通过融合来表示上采样并生成预测结果)
    - FeatureFusionBlock_custom：以torch的module为基础模型进行预测
- forward:传输数据到网络结构中。返回参数为out以及网络layer_4_rn
  - forward_vit

DPTDepthModel:定义DPT深度预测模型结构，并将该结构通过super函数传递给父类(DPT类)的initial函数进行初始化。该类的forward函数也是调用父类(DPT类)的forward函数
- Interpolate

输入：
path:.pt模型文件的相对路径。由class DPTDepthModel(DPT)通过load函数到class BaseModel(torch.nn.Module)，其中通过parameters = torch.load(path, map_location=torch.device('cpu'))来加载模型的参数文件
non_negative:有点不是很懂这个参数的含义
**kwargs：一些其他的参数比如backbone为主干网络的名称，由class DPTDepthModel(DPT)到class DPT(BaseModel)，通过make_encoder函数在hooks中选择对应的模型

输出：
forward函数的返回值为为out以及网络layer_4_rn，其中out为out = self.scratch.output_conv(path_1)函数的返回结果，layer_4_rn为self.scratch.layer4_rn(layer_4)函数的返回结果
