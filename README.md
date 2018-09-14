# F1-measure-for-semantic-segmentation
F1 measure for semantic segmentation

metric_f1.py 参数说明：

--predict_path 模型输出的mask
--predict_prefix 模型输出的mask的前缀，例如im12_crf.png, 其前缀为 im
--predict_format 模型输出的mask的类型，例如im12_crf.png, 其类型为 .png
--target_path groud truth mask
--target_prefix groud truth mask的前缀
--target_format groud truth mask的类型
--img_path 测试集图片
--img_prefix 测试集图片的前缀
--img_format 测试集图片的类型
--out_path 输出图片的路径
--out_prefix 输出图片的前缀
--out_format 输出图片的类型
--metric_file F1-measure的最大值、最小值、均值，以及在所有测试图片上的取值

详细使用，可参考 evaluation.sh

输出图片中，模型正确预测为前景的部分为原图片对应部分，模型正确预测为背景的部分为黑色，模型错误预测为前景的部分为红色，模型错误预测为背景的部分为蓝色。




