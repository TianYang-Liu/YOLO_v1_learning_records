import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self,S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        # 定义使用的损失函数
        self.mse=nn.MSELoss(reduction="sum")
        self.S=S
        self.B=B
        self.C=C
        # 定义参数
        self.lambda_coord=5
        self.lambda_noobj=0.5

    # 定义损失计算公式
    def forward(self,prediction,target):
        # 将prediction的结果reshape
        prediction = prediction.reshape(-1,self.S,self.S,self.C+self.B*5)
        # prediction的shape: (batch_size, 7, 7, 30)
        # target的shape: (batch_szie, 7, 7, 25)

        # 计算IOU
        # 计算第一个bbox的IOU

        iou_b1=intersection_over_union(prediction[...,21:25], target[...,21:25])# 0-19为类别概率，20为置信度，21-24为box坐标，25为置信度，26-29为bbox坐标
        iou_b2=intersection_over_union(prediction[...,26:30], target[...,21:25])
        # (batch_szie, 7, 7, 1)

        # 连接iou以比较大小,并且返回最大值的index
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)],dim=0)
        # (2,batch_szie, 7, 7, 1)

        iou_maxes, bestbox=torch.max(ious,dim=0) # bestbox是最大值的索引，它的值只能是0或1
        # (batch_szie, 7, 7, 1)


        # 判断是否有物体落入当前的box，值为0，1.对应论文中的Pr(Object)
        exists_box=target[...,20:21]

        # ===================== #
        #  FOR BOX COORDINATES  #
        # ===================== #

        # x,y
        box_predictions=exists_box*(
                (1-bestbox)*prediction[...,21:25]+bestbox*prediction[...,26:30]
        )
        # w,h
        box_predictions[...,2:4]=torch.sign(box_predictions[...,2:4])*torch.sqrt(torch.abs(box_predictions[...,2:4]+1e-6))


        box_targets = exists_box*target[...,21:25]

        # (N, 7, 7, 4)->(N,7*4*4)，no matter flatten成什么shape，只要保证box_predictions和box_target的shape相同即可，
        box_loss=self.mse(torch.flatten(box_predictions[...,21:25],start_dim=1),
                          torch.flatten(box_targets[...,21:25],start_dim=1))

        # ================================== #
        #    FOR OBJECT LOSS OF CONFIDENCE   #
        # ================================== #

        # 找到负责预测的box
        pred_box=(1-bestbox)*prediction[...,20:21]+bestbox*prediction[...,29:30]# confidence=Pr(Object)*IOU

        obj_loss=self.mse(torch.flatten(exists_box*pred_box),
                          torch.flatten(exists_box*target[...,20:21]))

        # ====================================== #
        #    FOR NO OBJECT LOSS OF CONFIDENCE    #
        # ====================================== #
        # 对于没有物体的cell，要对所有的（两个）bbox都进行计算损失。因为如果cell没有物体，就说明两个bbox都没有检测出物体。因此在没有物体的情况下cell两个bbox的loss都要计算
        # (N,S,S,1) -> (N,S*S) 无论怎么flatten，loss的值不会变
        noobj_loss=self.mse(torch.flatten((1 - exists_box)*prediction[...,20:21]), # 此时confidence=Pr(Object)
                            torch.flatten((1 - exists_box)*target[...,20:21]))
        noobj_loss+=self.mse(torch.flatten((1 - exists_box)*prediction[...,29:30]),
                            torch.flatten((1 - exists_box)*target[...,20:21]))

        # ==================== #
        #    FOR CLASS LOSS    #
        # ==================== #

        # (N,S,S,20) -> (N*S*S, 20) 无论怎么flatten，loss的值不会变
        cls_loss=self.mse(
            torch.flatten(exists_box*prediction[...,0:20],end_dim=-2),
            torch.flatten(exists_box*target[...,0:20],end_dim=-2)
        )

        loss=(self.lambda_coord*box_loss
              +obj_loss
              +self.lambda_noobj*noobj_loss
              +cls_loss)

        return loss