from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np

class Exp_Solar_Physics(Exp_Basic):
    """
    Special Experiment for Physics-Guided Solar Forecasting.
    Train on Residuals (IMFs), Test on Reconstructed Raw Power.
    """
    def __init__(self, args):
        super(Exp_Solar_Physics, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self, setting):
        # 标准训练流程，针对残差分量进行优化
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Predict Residuals (IMFs)
                outputs = self.model(batch_x, batch_x_mark, None, None)
                
                # 这里的 batch_y 是预处理后的 IMF 分量
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop: break
            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        # 验证集也只看残差 Loss，用于 Early Stopping
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        return np.average(total_loss)

    def test(self, setting, test=0):
        """
        ⚡️ 核心修改：物理重构测试
        """
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds_raw = [] # 重构后的总功率预测
        trues_raw = [] # 真实的原始总功率
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                
                # 1. 模型预测残差分量
                outputs_res = self.model(batch_x, batch_x_mark, None, None)
                outputs_res = outputs_res[:, -self.args.pred_len:, :]
                outputs_res = outputs_res.detach().cpu().numpy() # [B, L, V]
                
                # 2. 获取 Physics Baseline 和 Raw Truth
                # 我们利用 dataset 的 inverse_transform 或者直接从 raw data 读取
                # 这里假设 test_loader 的 dataset 有办法访问原始列
                # 为了简单，我们需要在 data_factory/loader 层面保证数据顺序
                
                # ⚠️ 关键 Hack: 
                # 我们假设 preprocessing 生成的 CSV 列顺序是: [date, P_RAW, P_PHY, IMF1...IMFn]
                # data_loader 在读取时，如果 features='M'，它会读取除 date 外的所有列。
                # 所以 batch_y 其实包含了 [P_RAW, P_PHY, IMFs...]
                # 假设 P_RAW 是第 0 列，P_PHY 是第 1 列，后面是 IMFs
                
                batch_y_full = batch_y.detach().cpu().numpy() # [B, L, All_Cols]
                
                p_raw_true = batch_y_full[:, -self.args.pred_len:, 0] # 真实总功率
                p_phy_base = batch_y_full[:, -self.args.pred_len:, 1] # 物理基线
                
                # 3. 物理重构: Pred_Raw = P_Phy + Sum(Pred_IMFs)
                # 注意：模型输出的是所有 features 的预测，我们需要只取 Power 相关的 IMF
                # 假设 Power 的 IMF 是第 2 列到第 2+K 列
                # 这里简化：假设模型预测了所有列，我们只把模型预测的 IMF 部分加起来
                # (实际操作中需根据你的列索引调整)
                
                # 假设模型输出的通道和 batch_y 的后几列对应（因为训练时输入也是所有列）
                # 这里简单地将预测的所有残差分量求和（需要你保证输入特征确实对应残差）
                # 如果输入包含温度分量，这里需要 mask 掉温度分量
                
                # 假设 Power Residuals 在模型输出的前 K 个通道
                k_modes = 8 # 你的 K
                pred_power_res_sum = np.sum(outputs_res[:, :, 2:2+k_modes], axis=2) 
                
                # 重构
                pred_total = p_phy_base + pred_power_res_sum
                pred_total = np.maximum(pred_total, 0) # ReLU Rectification
                
                preds_raw.append(pred_total)
                trues_raw.append(p_raw_true)

        preds_raw = np.concatenate(preds_raw, axis=0)
        trues_raw = np.concatenate(trues_raw, axis=0)
        
        # 4. 计算真实世界的 MSE/MAE
        mae, mse, rmse, mape, mspe = metric(preds_raw, trues_raw)
        print(f'✨ Physics-Reconstructed Test Result: MSE:{mse:.4f}, MAE:{mae:.4f}')
        
        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds_raw)
        
        return