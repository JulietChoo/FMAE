import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .abstract_class import AbstractEvaluator
import torch.distributed as dist
import os
from sklearn.metrics import roc_auc_score, roc_curve

class ImageEER(AbstractEvaluator):
    def __init__(self, threshold=0.5) -> None:
        self.name = "image-level EER"
        self.desc = "image-level EER"
        self.predict_label = torch.tensor([], device='cuda')
        self.label = torch.tensor([], device='cuda')
        self.cnt = torch.tensor(0, device='cuda')
        self.threshold = threshold
    
    def compute_eer(self, y_true, y_pred):
        # 计算ROC曲线的相关值
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        
        # 找到FPR和FRR（1 - TPR）最接近的那个阈值对应的索引
        diff = np.abs(fpr - (1 - tpr))
        idx = np.argmin(diff)
        
        # 计算EER
        eer = (fpr[idx] + (1 - tpr[idx])) / 2

        return eer.item()
    
    
    def batch_update(self, predict_label, label, *args, **kwargs):
        predict = predict_label.float().cuda()
        self.predict_label = torch.cat([self.predict_label, predict], dim=0)
        self.label = torch.cat([self.label, label], dim=0)
        self.cnt += torch.tensor(len(label), device='cuda')
        return None

    def epoch_update(self, **kwargs):
        if kwargs['distributed']:
            cnt = self.cnt.clone().detach().cuda()
            t_gather_cnt = [torch.zeros(1, dtype=torch.int64, device='cuda') for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(t_gather_cnt, cnt)
            
            max_cnt = torch.max(torch.stack(t_gather_cnt, dim=0), dim=0)[0].cuda()
            max_idx = torch.max(torch.stack(t_gather_cnt, dim=0), dim=0)[1].cuda()

            if max_cnt > self.cnt:
                self.predict_label = torch.cat([self.predict_label, torch.zeros(max_cnt-self.cnt, device='cuda')], dim=0)
                self.label = torch.cat([self.label, torch.zeros(max_cnt-self.cnt, device='cuda')], dim=0)

            t_label = self.label.float().cuda()
            t_predict_label = self.predict_label.float().cuda()

            t_gather_predict_label = [torch.zeros(max_cnt, dtype=torch.float32, device='cuda') for _ in range(dist.get_world_size())]
            t_gather_label = [torch.zeros(max_cnt, dtype=torch.float32, device='cuda') for _ in range(dist.get_world_size())]
            dist.barrier()

            dist.all_gather(t_gather_label, t_label)
            
            dist.barrier()
            dist.all_gather(t_gather_predict_label, t_predict_label)

            final_predict_label = torch.cat([t_gather_predict_label[idx][:cnt.item()] for idx, cnt in enumerate(t_gather_cnt)], dim=0).cuda()
            final_label = torch.cat([t_gather_label[idx][:cnt.item()] for idx, cnt in enumerate(t_gather_cnt)], dim=0).cuda()

            final_predict_label = final_predict_label.view(-1)
            final_label = final_label.view(-1)
            print(len(final_label))
            EER = self.compute_eer((final_label.detach().cpu().numpy()>self.threshold).astype(np.int32), final_predict_label.detach().cpu().numpy())
        else:
            EER = self.compute_eer((self.label.detach().cpu().numpy()>self.threshold).astype(np.int32), self.predict_label.detach().cpu().numpy())
        return EER
    
    def recovery(self):
        self.predict_label = torch.tensor([], device='cuda')
        self.label = torch.tensor([], device='cuda')
        self.cnt = torch.tensor(0, device='cuda')
    
    
def calculate_eer_sk(y_true, y_pred):
    # 计算ROC曲线的相关值
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # 找到FPR和FRR（1 - TPR）最接近的那个阈值对应的索引
    diff = np.abs(fpr - (1 - tpr))
    idx = np.argmin(diff)

    # 计算EER
    eer = (fpr[idx] + (1 - tpr[idx])) / 2
    return eer

def test_origin_image_eer():
    # test imageEER
    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://')
    
    num_gpus = torch.cuda.device_count()
    if dist.get_rank() == 0:
        print("number of GPUS", num_gpus)
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    DATA_LEN = 200
    float_tensor = torch.rand( DATA_LEN * num_gpus).cuda(local_rank)  # 生成一个长度为 5 的浮点数 tensor

    # 生成一个包含 0 或 1 的整数 tensor
    int_tensor = torch.randint(0, 2, (DATA_LEN * num_gpus,)).cuda(local_rank)
    # print(float_tensor)
    # print(int_tensor)
    
    evaluator = ImageEER(threshold=0.5)
    dist.barrier()
    dist.broadcast(float_tensor, src=0)
    dist.broadcast(int_tensor, src=0)
    # 收集所有的预测标签和真实标签，用于之后的 sklearn 验证
    all_predicts = []
    all_labels = []
    
    
    if dist.get_rank() != num_gpus - 1:
        idx = dist.get_rank() * DATA_LEN
        predict_labels = float_tensor[idx: idx + DATA_LEN].cuda(local_rank)
        true_labels = int_tensor[idx: idx + DATA_LEN].cuda(local_rank)
    else:
        idx = dist.get_rank() * DATA_LEN
        predict_labels = float_tensor[idx: idx + DATA_LEN-50].cuda(local_rank)
        true_labels = int_tensor[idx: idx + DATA_LEN-50].cuda(local_rank)


    if dist.get_rank() == 0:  # 只在 rank 0 进程中收集数据
        all_predicts = float_tensor.cpu().numpy()
        all_labels= int_tensor.cpu().numpy()
        # print(all_labels)
            

    # 运行 batch_update 更新统计数据
    evaluator.batch_update(predict_labels, true_labels)
    
    # 模拟一个 epoch 结束，调用 epoch_update 来计算 F1 分数
    gpu_eer_score = evaluator.epoch_update(distributed=True)
    if(dist.get_rank() == 0):
        print(f"AUC Score: {gpu_eer_score}")
        print(f"{calculate_eer_sk((all_labels[:-50]>0.5).astype(np.int32), (all_predicts[:-50]))}")


    # 清理分布式环境
    dist.destroy_process_group()


            



# # 示例用法和对比
if __name__ == "__main__":
    # 生成一些示例数据
    # batch_size, channels, height, width = 1, 1, 10, 10
    # predict = torch.rand(batch_size, channels, height, width)
    # mask = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    
    # # 生成一个 shape_mask
    # shape_mask = torch.randint(0, 2, (batch_size, channels, height, width)).float()

    # auc = PixelAUC()
    # reverse_auc = PixelAUC(mode="reverse")
    # double_auc = PixelAUC(mode="double")
    # # image_auc = Image_AUC()

    # auc_value_pytorch = auc.batch_update(predict, mask, shape_mask)
    # reverse_auc_value_pytorch = reverse_auc.batch_update(predict, mask, shape_mask)
    # double_auc_value_pytorch = double_auc.batch_update(predict, mask, shape_mask)
    # # image_auc_value_pytorch = image_auc(torch.tensor([[0.1],[0.3]]), torch.tensor([[1.],[0.]]))

    # # 转换为 numpy 数组用于 scikit-learn 计算
    # predict_np = (predict * shape_mask).flatten().numpy()
    # mask_np = (mask * shape_mask).flatten().numpy()

    # # 排除被 shape_mask 掩盖的部分
    # valid_mask_np = shape_mask.flatten().numpy() > 0
    # predict_np = predict_np[valid_mask_np]
    # mask_np = mask_np[valid_mask_np]

    # auc_value_sklearn = roc_auc_score(mask_np, predict_np)
    # reverse_auc_value_sklearn = roc_auc_score(mask_np, 1 - predict_np)
    # double_auc_value_sklearn = max(auc_value_sklearn, reverse_auc_value_sklearn)
    # # image_auc_value_sklearn = roc_auc_score(torch.tensor([[1],[0]]), torch.tensor([[0.1],[0.3]]))

    # print(f"PyTorch AUC: {auc_value_pytorch}")
    # print(f"scikit-learn AUC: {auc_value_sklearn}\n")

    # print(f"PyTorch Reverse AUC: {reverse_auc_value_pytorch}")
    # print(f"scikit-learn Reverse AUC: {reverse_auc_value_sklearn}\n")

    # print(f"PyTorch Double AUC: {double_auc_value_pytorch}")
    # print(f"scikit-learn Double AUC: {double_auc_value_sklearn}\n")

    # # print(f"PyTorch Image AUC: {image_auc_value_pytorch}")
    # # print(f"scikit-learn Image AUC: {image_auc_value_sklearn}")

    # os.environ['RANK'] = '0'  # 根据实际的进程排名设置  
    # os.environ['MASTER_ADDR'] = 'localhost'  # 或者是主节点的IP地址
    os.environ['MASTER_ADDR'] = 'localhost'  # 或者指定主节点的IP地址  
    os.environ['RANK'] = '0'  # 当前进程的rank
    os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())  
    os.environ['MASTER_PORT'] = '12345'  # 某个可用的端口
    test_origin_image_eer()
