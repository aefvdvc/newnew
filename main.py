import torch
import torch.utils.data as torch_data
from torchvision import datasets, transforms
from model import RoundtripModel, compute_roundtrip_gmm_loss, visualize_latent_space
from evaluate import evaluate
import argparse
import logging
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime  # 导入时间模块

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def get_dataset(dataset_name, transform, train=True):
    """
    Load the specified dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'MNIST', 'CIFAR10', 'FashionMNIST').
        transform (transforms.Compose): Transformations to apply to the dataset.
        train (bool, optional): Whether to load the training or test set. Defaults to True.

    Returns:
        datasets.VisionDataset: The loaded dataset.
    """
    if dataset_name == "MNIST":
        return datasets.MNIST('./data', train=train, download=False, transform=transform)
    elif dataset_name == "CIFAR10":
        return datasets.CIFAR10('./data', train=train, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        return datasets.FashionMNIST('./data', train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def save_reconstructed_images(model, data_loader, device, epoch, save_dir, dataset_name):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            z_enc, x_recon, _, _ = model(data)
            break  # 只取第一个批次的数据

    # 选择前16个图像进行可视化
    n_images = 16
    fig, axes = plt.subplots(2, n_images, figsize=(20, 4))
    for i in range(n_images):
        # 原始图像
        if dataset_name == "CIFAR10":
            axes[0, i].imshow(data[i].cpu().numpy().transpose(1, 2, 0))
        else:
            axes[0, i].imshow(data[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        # 重构图像
        if dataset_name == "CIFAR10":
            axes[1, i].imshow(x_recon[i].cpu().numpy().transpose(1, 2, 0))
        else:
            axes[1, i].imshow(x_recon[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')

    plt.savefig(os.path.join(save_dir, f'reconstructed_images_epoch_{epoch}.png'))
    plt.close()

def plot_loss_curve(losses, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(losses['total'], label='Total Loss')
    plt.plot(losses['loss_x'], label='Reconstruction Loss')
    plt.plot(losses['loss_kl'], label='KL Divergence Loss')
    plt.plot(losses['loss_rt'], label='Roundtrip Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

def save_model_with_timestamp(model, save_dir, epoch, loss=None):
    """
    保存模型并在文件名中加入时间戳和可选的额外信息。
    """
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 格式：YYYYMMDD_HHMMSS

    # 根据是否包含 loss 决定文件名格式
    if loss is not None:
        model_filename = f"model_epoch{epoch}_loss{loss:.4f}_{timestamp}.pth"
    else:
        model_filename = f"model_epoch{epoch}_{timestamp}.pth"

    # 生成保存路径
    model_path = os.path.join(save_dir, model_filename)

    # 保存模型状态字典
    torch.save(model.state_dict(), model_path)

    logger.info(f"Model saved to {model_path}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Roundtrip Model Training and Evaluation")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--latent_dim", type=int, default=20, help="Dimension of the latent space")
    parser.add_argument("--n_components", type=int, default=10, help="Number of components in GMM")
    parser.add_argument("--input_dim", type=int, default=28 * 28, help="Input dimension (for MNIST)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset to use (MNIST, CIFAR10, FashionMNIST)")
    parser.add_argument("--save_model", action="store_true", help="Save the trained model")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to save or load the model")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据预处理
    if args.dataset == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])

    # 加载数据集
    train_dataset = get_dataset(args.dataset, transform, train=True)
    train_loader = (torch_data
                    .DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True))

    # 初始化模型、优化器和设备
    device = torch.device(args.device)
    model = RoundtripModel(input_dim=args.input_dim, latent_dim=args.latent_dim, n_components=args.n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练模型
    model.train()
    losses = {'total': [], 'loss_x': [], 'loss_kl': [], 'loss_rt': []}
    start_time = time.time()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_loss_x = 0.0
        total_loss_kl = 0.0
        total_loss_rt = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            try:
                loss, loss_x, loss_kl, loss_rt = compute_roundtrip_gmm_loss(data, model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_loss_x += loss_x.item()
                total_loss_kl += loss_kl.item()
                total_loss_rt += loss_rt.item()
            except Exception as e:
                logger.error(f"Error during training: {e}")
                raise

        avg_loss = total_loss / len(train_loader)
        avg_loss_x = total_loss_x / len(train_loader)
        avg_loss_kl = total_loss_kl / len(train_loader)
        avg_loss_rt = total_loss_rt / len(train_loader)

        losses['total'].append(avg_loss)
        losses['loss_x'].append(avg_loss_x)
        losses['loss_kl'].append(avg_loss_kl)
        losses['loss_rt'].append(avg_loss_rt)

        logger.info(f"Epoch {epoch + 1}/{args.epochs}, Total Loss: {avg_loss:.4f}, Reconstruction Loss: {avg_loss_x:.4f}, KL Divergence Loss: {avg_loss_kl:.4f}, Roundtrip Loss: {avg_loss_rt:.4f}")

        # 保存重构图像
        save_reconstructed_images(model, train_loader, device, epoch, args.save_dir, args.dataset)

        # 保存模型
        if args.save_model:
            save_model_with_timestamp(model, args.save_dir, epoch, avg_loss)

    # 可视化潜在空间
    visualize_latent_space(model, train_loader, device)

    # 绘制损失曲线
    plot_loss_curve(losses, args.save_dir)

    # 评估模型
    nmi, ari, acc = evaluate(model, train_loader, device)
    logger.info(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    logger.info(f"Adjusted Rand Index (ARI): {ari:.4f}")
    logger.info(f"Clustering Accuracy (ACC): {acc:.4f}")

    # 记录训练时间
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

if __name__ == "__main__":
    main()