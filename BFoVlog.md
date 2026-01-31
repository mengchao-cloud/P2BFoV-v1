point-to-bfov开发日志

2025/12/22:
    目标完成正提案初始生成任务，
    原始函数def gen_proposals_from_cfg(gt_points, proposal_cfg, img_meta):
    改写的思路就是沿着训练器的数据流向进行改写，针对当前目标会搁置一些问题，放到下面后续会进行处理
        1、为了方便后续任务，这里的所有bbox都不对其进行更改将其视为bfov，进而快速梳理
        2、主训练流程里面有特征提取是需要改写成球面的特征提取思路，搁置
        3、gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]这里尚未弄清，搁置
        4、中心点坐标生成这一步，由于需要数据集里面的内容，后续需要针对数据读取流程进行处理，搁置
        5、#阶段0的初始提案生成，这里需要改写函数使之生成球面BFoV，当前任务需求/////////////
        6、需要处理清楚图像元信息的字段里面的信息是什么，搁置
        7、准备将提案范围改为base_scales=[5, 10, 20, 40, 90, 180]单位/度，搁置


    改写过程中的思路：
        1、由于球面提案的生成不会超越界限，实际的提案生成唯一的约束就是bfov的水平和垂直视场不能超过180度
        2、原始框架的基础尺度是base = min(img_w, img_h) / 100，
            提案尺度为base_scales=[4, 8, 16, 32, 64, 128],
            改动为base=1度，base_scales=[5, 10, 20, 40, 90, 180]
2025/12/23: 
    解决中心抖动的时候的经纬度变换问题（所有表示都是弧度计算）

    改写过程中的思路
        1、原始框架的中心抖动是沿着水平和垂直方向进行抖动的，
        2、但是在球面坐标下，水平方向的抖动会导致经度和纬度的同时变化，而且向左向右抖动的时候最终的纬度是一样的，经度是关于原始经度对称的，垂直方向的抖动会导致纬度的变化不会改变经度
2025/12/24: 
    目标：完成def gen_proposals_from_cfg(gt_points, proposal_cfg, img_meta)改写
    1、配置文件里面不需要cut_mode。后面需要删去，搁置
    2、这里需要重写这个函数base_proposals = bbox_cxcywh_to_xyxy(base_proposals)
        但是又不知道是否用得到，按照球面的iou计算的时候或许不需要这个。  搁置

    改写思路：
        1、抖动之后的中心坐标需要进行约束，这里做了一个约束函数def constrain_spherical_coords(centers):
        2、原始框架设置的对称裁剪模式会根据中心坐标对提案进行裁剪，球面上不需要
2025/12/28: 


    改动思路：
        1、原始框架是按照左上角和右下角的思路进行的box显示，但是在bfov里面我们按照中心坐标和bfov的角度进行显示
        2、所以提案生成后只要范围以及坐标合理即可，故不需要裁剪吗，由于原框架需要有效提案张量，因而我们选择不裁剪的格式，
            尽可能不改动原生框架
        3、最终输出两个张量，
            base_proposal_list张量形状为[N, 5, 4]，N是批量大小，5是每个中心抖动后的5个提案，4是提案信息[x1, y1, w1, h1]
            proposals_valid_list形状为[N, 5, 1]，N是批量大小，5是每个中心对应5个提案，1是是否有效（1表示有效，0表示无效）

2025/12/29: 
    目标对负提案生成函数进行改写def gen_negative_proposals(gt_points, proposal_cfg, aug_generate_proposals, img_meta):
    改写搁置;
        1、iou = bbox_overlaps(neg_bboxes, pos_box)这里需要重新改写iou计算，而且这里可能采用的是torch方式因为不只计算iou，
            而是计算所有正负提案之间的iou，搁置
        2、原框架为了防止负提案过于接近正提案，设置了一个gt_min_box = torch.cat([gt_point - 10, gt_point + 10], dim=1)
            基于每个点生成的最小box，问题是这个并没有在该函数中使用，或许有别的用处，搁置

    改写思路：
        1、原框架生成负提案的思路是随机生成500（gen_num_neg）个左上角右下角的负提案，
            改写方式为随机生成500（gen_num_neg）个中心坐标和bfov的角度
        2、
2025/12/30: 
    目标完成精细提案的生成函数def gen_fine_proposals(gt_points, proposal_cfg, img_meta):
    改写搁置;
        1、gen_mode = fine_proposal_cfg['gen_proposal_mode']，这个是精细提案的生成模式，
            有两种模式，一种是随机生成，一种是基于bfov的角度生成
    改写思路
        1、原框架使用配置文件来调整不同的精细提案生成，后续这个可以改成笛卡尔加权或球面加权
        2、球面生成的提案不需要裁剪

2026/1/2: 
    完成边界循环逻辑，也就是循环padding的逻辑，和精细bfov的生成函数
    1、骨干层和颈部的配置文件都需要添加这个配置conv_cfg=dict(type='Conv', padding_mode='circular')  由于目前只对检测器主文件和head文件进行改动，因而这个搁置
    2、需要注意除了颈部或者骨干两个部分需要调整卷积padding以外，rpn或者roihead可能也需要，后面碰到需要处理，这里暂时不处理，搁置
    3、base_boxes_ = bbox_xyxy_to_cxcywh(base_boxes)精细提案生成部分需要这个，暂时搁置



    改写思路：
        需要改写卷积padding的部分：有预处理步骤、骨干层、颈部网络
        base_boxes_ = bbox_xyxy_to_cxcywh(base_boxes)，平面上使用两点式来表示目标提案，现在需要对这部分进行确定，
            查看其他代码是如何实现的尤其关注如何确定特征跟提案的归属问题
2026/1/10: 
    目标：完成左右掩码生成函数generate_bfov_masks,并且调整为加速计算


截至到此基本将整个流程按照全景的思路改完一遍，包括生成正负提案精细提案球面iou，以及加权融合掩码特征提取，后面准备再按照数据集输入的流程通一遍，因为有些地方需要数据集的内容确认

 2026/1/17: 
    目标：把整个流程通一遍
    下面这个评估可能需要调整一下配置，暂时搁置
            evaluation = dict(
            interval=12,  # 评估间隔（每12个epoch）
            metric='bbox',  # 评估指标为边界框
            save_result_file=work_dir + '_' + str(test_scale) + '_latest_result.json',  # 结果保存文件
            do_first_eval=False,  # 不进行首次评估
            do_final_eval=True,  # 进行最终评估
        )
2026-01-22:
    已经将有关的训练推理评估流程逻辑全部改写完毕，并且梳理了一遍，
    目标：进行运行测试相关指令的走向梳理并尝试用本地gpu测试是否能跑通

2026-01-31:
    尺度分类为：0.5、1、2
    iou计算改成sphiou_efficient_pop

    后面放弃了尺度分类选择特征层，这里的逻辑不通，改用sph_loss做增量

    