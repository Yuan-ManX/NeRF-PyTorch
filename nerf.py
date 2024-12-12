import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


@torch.no_grad()
def test(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    """
    Args:
        hn: near plane distance
        hf: far plane distance
        dataset: dataset to render
        chunk_size (int, optional): chunk size for memory efficiency. Defaults to 10.
        img_index (int, optional): image index to render. Defaults to 0.
        nb_bins (int, optional): number of bins for density estimation. Defaults to 192.
        H (int, optional): image height. Defaults to 400.
        W (int, optional): image width. Defaults to 400.
        
    Returns:
        None: None
    """
    """
    测试函数，用于渲染指定图像。

    参数:
        hn (float): 近平面距离。
        hf (float): 远平面距离。
        dataset (Tensor): 用于渲染的数据集，形状为 (N, 6)，前 3 维为射线起点，后 3 维为射线方向。
        chunk_size (int, 可选): 分块大小，用于内存效率。默认值为 10。
        img_index (int, 可选): 要渲染的图像索引。默认值为 0。
        nb_bins (int, 可选): 密度估计的箱子数量。默认值为 192。
        H (int, 可选): 图像高度。默认值为 400。
        W (int, 可选): 图像宽度。默认值为 400。

    返回:
        None: 无返回值。
    """
    # 从数据集中提取射线起点和射线方向
    # 射线起点，形状为 (H*W, 3)
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    # 射线方向，形状为 (H*W, 3)
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    
    # 存储渲染后的像素值
    data = []   # list of regenerated pixel values

    # 遍历图像的高度分块
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks # 使用 ceil 确保覆盖所有行
        # Get chunk of rays
        # 获取当前分块的射线起点和射线方向
        # 当前分块的射线起点，形状为 (chunk_size*W, 3)
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        # 当前分块的射线方向，形状为 (chunk_size*W, 3)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)    

        # 使用渲染函数渲染当前分块的像素值    
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        # 将渲染后的像素值添加到数据列表中
        data.append(regenerated_px_values)

    # 将所有分块的渲染结果拼接起来，并重塑为图像形状
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3) # 重塑为 (H, W, 3) 的图像
    
    # 显示和保存渲染的图像
    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()

    # 提示用户渲染完成
    print(f'Image {img_index} rendered and saved.')


class NerfModel(nn.Module):
    """
    NeRF 模型，用于生成神经辐射场（Neural Radiance Fields）。

    参数:
        embedding_dim_pos (int, 可选): 位置编码的维度。默认值为 10。
        embedding_dim_direction (int, 可选): 方向编码的维度。默认值为 4。
        hidden_dim (int, 可选): 隐藏层的维度。默认值为 128。
    """
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):   
        super(NerfModel, self).__init__()
        
        # 第一个全连接块，用于处理位置编码和方向编码的组合输入
        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(), # 输入维度为位置编码维度 * 6 + 3
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        # density estimation
        # 密度估计块，用于预测体密度
        self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU(), # 输入维度为位置编码维度 * 6 + 隐藏层维度 + 3
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), ) # 输出维度为隐藏层维度 + 1（体密度）
        # color estimation
        # 颜色估计块，用于预测颜色
        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), # 输入维度为方向编码维度 * 6 + 隐藏层维度 + 3
            nn.ReLU(), 
            )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), # 输出维度为 3（RGB 颜色）
                                    nn.Sigmoid(),  # 使用 Sigmoid 激活函数将输出限制在 [0, 1] 范围内
                                    )

        # 存储位置编码维度
        self.embedding_dim_pos = embedding_dim_pos
        # 存储方向编码维度
        self.embedding_dim_direction = embedding_dim_direction
        # 存储 ReLU 激活函数
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        """
        对输入张量 x 进行位置编码。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, 3)。
            L (int): 位置编码的层数。

        返回:
            Tensor: 位置编码后的张量，形状为 (batch_size, embedding_dim_pos * 6)。
        """
        # 初始化输出列表，包含原始输入
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x)) # 添加正弦编码
            out.append(torch.cos(2 ** j * x)) # 添加余弦编码
        # 将所有编码拼接起来，形状为 (batch_size, embedding_dim_pos * 6)
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        """
        前向传播方法，预测颜色和体密度。

        参数:
            o (Tensor): 射线起点，形状为 (batch_size, 3)。
            d (Tensor): 射线方向，形状为 (batch_size, 3)。

        返回:
            Tuple[Tensor, Tensor]: 返回预测的颜色和体密度。
                - c (Tensor): 预测的颜色，形状为 (batch_size, 3)。
                - sigma (Tensor): 预测的体密度，形状为 (batch_size,)。
        """
        # 对射线起点进行位置编码，形状为 (batch_size, embedding_dim_pos * 6)
        emb_x = self.positional_encoding(o, self.embedding_dim_pos) # emb_x: [batch_size, embedding_dim_pos * 6]
        # 对射线方向进行位置编码，形状为 (batch_size, embedding_dim_direction * 6)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction) # emb_d: [batch_size, embedding_dim_direction * 6]
        
        # 通过第一个全连接块处理位置编码，输出形状为 (batch_size, hidden_dim)
        h = self.block1(emb_x) # h: [batch_size, hidden_dim]
        # 将隐藏层输出和位置编码拼接起来，输入到密度估计块，输出形状为 (batch_size, hidden_dim + 1)
        tmp = self.block2(torch.cat((h, emb_x), dim=1)) # tmp: [batch_size, hidden_dim + 1]

        # 分离隐藏层输出和体密度，sigma 为体密度，形状为 (batch_size,)
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1]) # h: [batch_size, hidden_dim], sigma: [batch_size]
        # 将隐藏层输出和方向编码拼接起来，输入到颜色估计块，输出形状为 (batch_size, hidden_dim // 2)
        h = self.block3(torch.cat((h, emb_d), dim=1)) # h: [batch_size, hidden_dim // 2]
        # 通过颜色估计块预测颜色，输出形状为 (batch_size, 3)
        c = self.block4(h) # c: [batch_size, 3]

        # 返回预测的颜色和体密度
        return c, sigma


def compute_accumulated_transmittance(alphas):
    """
    计算累积透射率。

    参数:
        alphas (Tensor): 不透明度张量，形状为 (batch_size, nb_bins)。

    返回:
        Tensor: 累积透射率，形状为 (batch_size, nb_bins)。
    """
    # 计算累积乘积，得到累积透射率
    accumulated_transmittance = torch.cumprod(alphas, 1)
    # 在累积透射率前面添加一列全 1，用于处理第一个时间步的透射率
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    """
    渲染射线，返回每个射线的颜色。

    参数:
        nerf_model (nn.Module): NeRF 模型，用于预测颜色和不透明度。
        ray_origins (Tensor): 射线起点，形状为 (batch_size, 3)。
        ray_directions (Tensor): 射线方向，形状为 (batch_size, 3)。
        hn (float, 可选): 近平面距离。默认值为 0.0。
        hf (float, 可选): 远平面距离。默认值为 0.5。
        nb_bins (int, 可选): 深度采样的箱子数量。默认值为 192。

    返回:
        Tensor: 渲染后的像素颜色，形状为 (batch_size, 3)。
    """
    device = ray_origins.device

    # 生成深度采样点
    # 在近平面和远平面之间均匀采样深度，形状为 (batch_size, nb_bins)
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)

    # Perturb sampling along each ray.
    # 对每个射线的采样点进行扰动，增加随机性
    # 计算相邻采样点的中点，形状为 (batch_size, nb_bins - 1)
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    # 计算下界，形状为 (batch_size, nb_bins)
    lower = torch.cat((t[:, :1], mid), -1)
    # 计算上界，形状为 (batch_size, nb_bins)
    upper = torch.cat((mid, t[:, -1:]), -1)
    # 生成均匀随机数，形状为 (batch_size, nb_bins)
    u = torch.rand(t.shape, device=device)

    # 对采样点进行扰动，形状为 (batch_size, nb_bins)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]

    # 计算相邻采样点之间的距离，形状为 (batch_size, nb_bins)
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    # 计算每个采样点的 3D 坐标
    # 计算 3D 坐标，形状为 (batch_size, nb_bins, 3)
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    # 将射线方向张量扩展到与 x 相同的形状
    # 扩展射线方向，形状为 (batch_size, nb_bins, 3)
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1) 
    
    # 使用 NeRF 模型预测颜色和不透明度
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    # 重塑颜色张量形状为 (batch_size, nb_bins, 3)
    colors = colors.reshape(x.shape)
    # 重塑不透明度张量形状为 (batch_size, nb_bins)
    sigma = sigma.reshape(x.shape[:-1])

    # 计算不透明度，形状为 (batch_size, nb_bins)
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    # 计算权重，形状为 (batch_size, nb_bins, 1)
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    # Compute the pixel values as a weighted sum of colors along each ray
    # 计算像素颜色为每个射线上的颜色加权求和
    # 计算像素颜色，形状为 (batch_size, 3)
    c = (weights * colors).sum(dim=1)
    # 计算权重之和，用于正则化白色背景
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background 

    # 返回渲染后的像素颜色
    return c + 1 - weight_sum.unsqueeze(-1)


def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192, H=400, W=400):
    """
    训练 NeRF 模型。

    参数:
        nerf_model (nn.Module): 要训练的 NeRF 模型。
        optimizer (torch.optim.Optimizer): 优化器，用于更新模型参数。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器，用于调整学习率。
        data_loader (torch.utils.data.DataLoader): 数据加载器，用于加载训练数据。
        device (str, 可选): 使用的设备，'cpu' 或 'cuda'。默认值为 'cpu'。
        hn (float, 可选): 近平面距离。默认值为 0.0。
        hf (float, 可选): 远平面距离。默认值为 1.0。
        nb_epochs (int, 可选): 训练的轮数。默认值为 100,000。
        nb_bins (int, 可选): 深度采样的箱子数量。默认值为 192。
        H (int, 可选): 图像高度。默认值为 400。
        W (int, 可选): 图像宽度。默认值为 400。

    返回:
        List[float]: 训练损失列表。
    """
    # 初始化训练损失列表
    training_loss = []
    # 遍历训练的每一轮
    for _ in tqdm(range(nb_epochs)):
        # 遍历数据加载器中的每个批次
        for batch in data_loader:
            # 提取射线起点，并移动到指定设备，形状为 (batch_size, 3)
            ray_origins = batch[:, :3].to(device)
            # 提取射线方向，并移动到指定设备，形状为 (batch_size, 3)
            ray_directions = batch[:, 3:6].to(device)
            # 提取真实像素值，并移动到指定设备，形状为 (batch_size, 3)
            ground_truth_px_values = batch[:, 6:].to(device)
            
            # 使用 NeRF 模型渲染像素值
            # 渲染像素值，形状为 (batch_size, 3)
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins) 
            # 计算损失（均方误差）
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            # 清零优化器的梯度
            optimizer.zero_grad()
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()

            # 将损失值添加到训练损失列表中
            training_loss.append(loss.item())

        # 更新学习率
        scheduler.step()

        # 对测试集中的每个图像进行测试渲染
        for img_index in range(200):
            test(hn, hf, testing_dataset, img_index=img_index, nb_bins=nb_bins, H=H, W=W)
    
    # 返回训练损失列表
    return training_loss



if __name__ == '__main__':
    device = 'cuda'
    
    # 加载训练数据
    # 从 'training_data.pkl' 文件中加载训练数据，并转换为 PyTorch 张量
    training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))

    # 加载测试数据
    # 从 'testing_data.pkl' 文件中加载测试数据，并转换为 PyTorch 张量
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))

    # 创建 NeRF 模型，设置隐藏层维度为 256，并将模型移动到指定设备
    model = NerfModel(hidden_dim=256).to(device)
    # 创建 Adam 优化器，设置学习率为 5e-4，优化模型的所有参数
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # 创建 MultiStepLR 学习率调度器，在第 2、4、8 个 epoch 时将学习率乘以 0.5
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

    # 创建数据加载器，加载训练数据集，设置批次大小为 1024，并打乱数据
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)

    # 训练模型
     # 调用 train 函数训练模型，设置训练轮数为 16，设备为 'cuda'，近平面距离为 2，远平面距离为 6，深度采样箱子数量为 192，图像高度和宽度均为 400
    train(model, model_optimizer, scheduler, data_loader, nb_epochs=16, device=device, hn=2, hf=6, nb_bins=192, H=400,
          W=400)
