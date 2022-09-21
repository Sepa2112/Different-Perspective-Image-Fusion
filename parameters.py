from torchvision import transforms


class Paras:
    # my tap-top
    txt_train_source = './data/train/source.txt'
    txt_train_target = './data/train/target.txt'
    txt_train_labels = './data/train/label.txt'

    txt_valid_source = './data/valid_sar/source.txt'
    txt_valid_target = './data/valid_sar/target.txt'
    txt_valid_labels = './data/valid_sar/label.txt'

    txt_test_source = './data/pre/py/source.txt'
    txt_test_target = './data/pre/py/target.txt'
    txt_test_labels = './data/pre/py/label.txt'

    summary_writer_location = 'RunWithValid/'

    '''训练文件参数'''
    gpus = [3]
    batch_size = 1  # 最好为1
    num_workers = 4
    is_use_cudnn = True

    '''优化器'''
    lr = 1e-4
    optimizer_adam_beta = 0.9  # [0.5, 0.9]

    '''学习率衰减'''
    scheduler = "StepLR"  # StepLR, MultiStepLR
    scheduler_step_size = 20
    milestones = []
    scheduler_gamma = 0.75

    start_epoch = 1
    epochs = 300
    interval_show_step = 5  # 显示的step
    interval_save_epoch = 1  # 存储模型的epoch间隔

    '''损失函数相关'''
    k_rec = 1  # 重构损失系数
    k_warp = 3  # 中间层仿射变换损失系数
    k_reg = 0.005  # 正则化项系数
    k_ssim = 0.86 * 0.2
    k_l1 = 0.14 * 0.2
    k_l2 = 0.2

    model_pfn_path = "./models/PixelFusionNet_145.pth"  # 生成器模型路径
    initial_pfn = None  # 是否初始化生成模型
    loss_lowest_valid = 20

    ver_per = 0.1  # 感知损失网络的版本选择，[0.0, 0.1]
    net_per = 'vgg16'  # 感知损失网络类型选择 [vgg16, vgg19]

    '''测试文件参数'''
    test_result_path = "result_images"
    input_size_test = 512

    trainTransform = transforms.Compose([transforms.ToTensor()])
    validTransform = transforms.Compose([
        # transforms.Resize(256),
        transforms.ToTensor()
    ])
